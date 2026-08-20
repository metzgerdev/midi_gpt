"""Iterated DPO: the distribution must move toward the edits and stay inside the KL budget.

Slow relative to test_inference.py — it runs five real training rounds — but it covers the
thing unit tests cannot: that repeated fine-tuning converges instead of running away. The
pre-fix code passes the "moves" half and fails the "contained" half.

Pairs are synthesised rather than read from training_data/dpo/, so the test does not depend
on run folders, drum loops, or a mined corpus.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from model.checkpoints import load_note_model
from utils.chords import progression_to_track
from utils.config import (
    CKPT_DIR, NOTE_BOS, NOTE_EOS, NOTE_MIDI_HI, NOTE_MIDI_LO, NOTE_PITCH0, NOTE_REST,
    NOTE_STEPS, NOTE_SUSTAIN,
)
from utils.device import pick_device
from train.finetune_dpo import anchor_contexts, kl_to_anchor, to_loader, train_dpo
from model.note_model import generate_notes, note_chord_cond, note_grid_cond
from utils.scoring import fit_and_lock

ROUNDS = 5
SECTIONS = 4
EPOCHS = 4
KL_TARGET = 0.02
MAX_SUSTAIN = 1              # the taught edit: every note one step long
BARS, STEPS_PER_BAR = 4, 16
PROGRESSIONS = ["Am-F-C-G", "Cm-G#-Fm-Gm", "Em-C-Am-Bm", "Gm-D#-Cm-Dm"]


def four_on_the_floor() -> np.ndarray:
    grid = np.zeros(NOTE_STEPS, np.float32)
    grid[::4] = 1.0
    return grid


def sample_section(model, progression: str, seed: int, device) -> np.ndarray:
    torch.manual_seed(seed)
    chroma = progression_to_track(progression, NOTE_STEPS)
    out = generate_notes(model, note_grid_cond(four_on_the_floor()), NOTE_BOS, NOTE_EOS,
                         max_steps=NOTE_STEPS, temperature=1.0, top_p=0.98,
                         chord_tok=note_chord_cond(chroma), device=device)
    tokens = out[0, 1:].cpu().numpy()
    return np.pad(tokens, (0, max(0, NOTE_STEPS - len(tokens))),
                  constant_values=NOTE_REST)[:NOTE_STEPS]


def shorten_notes(tokens: np.ndarray) -> np.ndarray:
    """The synthetic taste signal: no note is held past MAX_SUSTAIN steps.

    Chosen because it is unambiguously measurable — mean note length has to fall if the
    model absorbed it — and because it needs no information the conditioning withholds.
    Forcing chord roots would not work as a probe: the chroma is a commutative bag with no
    root marked, so the model can only partly learn it.
    """
    out = tokens.copy()
    held = 0
    for step, token in enumerate(out):
        if token >= NOTE_PITCH0:
            held = 1
        elif token == NOTE_SUSTAIN:
            held += 1
            if held > MAX_SUSTAIN:
                out[step] = NOTE_REST
                held = 0
        else:
            held = 0
    return out


def transposed_pair(chosen, rejected, chroma, grid_tok, shift):
    """One (chosen, rejected) pair in one key — notes and chroma rotate together."""
    def sequence(tokens):
        out = tokens.copy()
        voiced = out >= NOTE_PITCH0
        midi = NOTE_MIDI_LO + out[voiced] - NOTE_PITCH0 + shift
        out[voiced] = NOTE_PITCH0 + np.clip(midi, NOTE_MIDI_LO, NOTE_MIDI_HI) - NOTE_MIDI_LO
        return np.concatenate([[NOTE_BOS], out, [NOTE_EOS]]).astype(np.int64)

    chord_tok = note_chord_cond(np.roll(chroma, shift, axis=1).astype(np.float32))
    take, leave = sequence(chosen), sequence(rejected)
    return ((take[:-1], take[1:], grid_tok, chord_tok),
            (leave[:-1], leave[1:], grid_tok, chord_tok))


def synthetic_pairs(policy, round_index, device):
    """Sample from the current policy, shorten the notes, expand across twelve keys."""
    grid_tok = note_grid_cond(four_on_the_floor())
    pairs = []
    for i in range(SECTIONS):
        progression = PROGRESSIONS[i % len(PROGRESSIONS)]
        chroma = progression_to_track(progression, NOTE_STEPS)
        rejected = sample_section(policy, progression, 977 * round_index + 13 * i, device)
        chosen = shorten_notes(rejected)
        if np.array_equal(chosen, rejected):
            continue
        pairs += [transposed_pair(chosen, rejected, chroma, grid_tok, s) for s in range(12)]
    return pairs


def distribution(model, device, n_seeds=4):
    """Mean note length and chord fit over a fixed evaluation sweep."""
    grid = four_on_the_floor()
    lengths, fits = [], []
    for progression in PROGRESSIONS:
        chroma = progression_to_track(progression, NOTE_STEPS)
        for seed in range(n_seeds):
            tokens = sample_section(model, progression, 50_000 + seed * 31, device)
            run = 0
            for token in tokens:
                if token >= NOTE_PITCH0:
                    if run:
                        lengths.append(run)
                    run = 1
                elif token == NOTE_SUSTAIN and run:
                    run += 1
                else:
                    if run:
                        lengths.append(run)
                    run = 0
            if run:
                lengths.append(run)
            fits.append(fit_and_lock(tokens, chroma, grid)[0])
    return {"mean_note_len": float(np.mean(lengths)) if lengths else 0.0,
            "chord_fit": float(np.mean(fits))}


@pytest.mark.slow
def test_iterated_dpo_moves_the_distribution_and_stays_inside_the_kl_budget():
    checkpoint = CKPT_DIR / "bass_notes_gpt.pt"
    if not checkpoint.exists():
        pytest.skip(f"local inference asset is absent: {checkpoint}")

    device = pick_device("cpu")
    torch.manual_seed(0)
    policy = load_note_model(checkpoint, device)
    anchor = copy.deepcopy(policy).eval().requires_grad_(False)

    before = distribution(policy, device)
    kl, lam = 0.0, 1.0
    for round_index in range(1, ROUNDS + 1):
        pairs = synthetic_pairs(policy, round_index, device)
        assert pairs, f"round {round_index} produced no pairs to learn from"
        contexts = anchor_contexts(anchor, pairs, 16, device, seed=round_index)
        kl, lam = train_dpo(policy, anchor,
                            to_loader(pairs, 16, shuffle=True, seed=round_index),
                            contexts, beta=0.1, lr=1e-5, epochs=EPOCHS,
                            kl_target=KL_TARGET, kl_lambda=lam, device=device)
    after = distribution(policy, device)

    # 1. the taught signal moved: shorter notes, in the direction of the edit
    assert after["mean_note_len"] < before["mean_note_len"], (
        f"notes did not shorten: {before['mean_note_len']:.3f} -> {after['mean_note_len']:.3f}"
    )

    # 2. drift stayed inside the budget. Without the fixed anchor and KL term this is where
    #    it fails: the margin runs away and KL climbs without limit.
    with torch.no_grad():
        final_kl = float(kl_to_anchor(policy, anchor_contexts(anchor, pairs, 16, device, 0)))
    assert kl <= KL_TARGET * 3.0, f"KL {kl:.4f} ran away from budget {KL_TARGET}"
    assert final_kl <= KL_TARGET * 3.0, (
        f"KL on fresh contexts {final_kl:.4f} exceeds 3x the {KL_TARGET} budget"
    )

    # 3. it stayed a usable model rather than trading everything for the edit
    assert after["chord_fit"] > before["chord_fit"] * 0.85, (
        f"chord fit collapsed: {before['chord_fit']:.3f} -> {after['chord_fit']:.3f}"
    )


@pytest.mark.slow
def test_fixed_anchor_bounds_drift_better_than_re_anchoring():
    """The other half of the fix, isolated.

    With the KL term on, its controller clamps both strategies and hides the difference,
    so this runs with KL disabled. Re-anchoring to the previous round bounds each round
    against the last one and lets the total compound; a fixed anchor bounds the total.
    """
    checkpoint = CKPT_DIR / "bass_notes_gpt.pt"
    if not checkpoint.exists():
        pytest.skip(f"local inference asset is absent: {checkpoint}")

    device = pick_device("cpu")
    drift = {}
    for label in ("fixed", "rolling"):
        torch.manual_seed(0)
        policy = load_note_model(checkpoint, device)
        origin = copy.deepcopy(policy).eval().requires_grad_(False)
        for round_index in range(1, ROUNDS + 1):
            pairs = synthetic_pairs(policy, round_index, device)
            anchor = (origin if label == "fixed"
                      else copy.deepcopy(policy).eval().requires_grad_(False))
            train_dpo(policy, anchor, to_loader(pairs, 16, shuffle=True, seed=round_index),
                      None, beta=0.1, lr=1e-5, epochs=EPOCHS,
                      kl_target=0.0, kl_lambda=1.0, device=device)
        with torch.no_grad():                       # always measured against the origin
            drift[label] = float(kl_to_anchor(policy, anchor_contexts(origin, pairs, 16,
                                                                      device, seed=0)))

    assert drift["fixed"] < drift["rolling"], (
        f"re-anchoring did not drift further: fixed={drift['fixed']:.4f} "
        f"rolling={drift['rolling']:.4f}"
    )


@pytest.mark.slow
def test_kl_term_is_what_bounds_the_drift():
    """The control: same rounds, KL disabled, and the drift is materially larger."""
    checkpoint = CKPT_DIR / "bass_notes_gpt.pt"
    if not checkpoint.exists():
        pytest.skip(f"local inference asset is absent: {checkpoint}")

    device = pick_device("cpu")
    kls = {}
    for label, kl_target in (("kl_on", KL_TARGET), ("kl_off", 0.0)):
        torch.manual_seed(0)
        policy = load_note_model(checkpoint, device)
        anchor = copy.deepcopy(policy).eval().requires_grad_(False)
        lam = 1.0
        for round_index in range(1, ROUNDS + 1):
            pairs = synthetic_pairs(policy, round_index, device)
            contexts = (anchor_contexts(anchor, pairs, 16, device, seed=round_index)
                        if kl_target else None)
            _, lam = train_dpo(policy, anchor,
                               to_loader(pairs, 16, shuffle=True, seed=round_index),
                               contexts, beta=0.1, lr=1e-5, epochs=EPOCHS,
                               kl_target=kl_target, kl_lambda=lam, device=device)
        with torch.no_grad():
            probe = anchor_contexts(anchor, pairs, 16, device, seed=0)
            kls[label] = float(kl_to_anchor(policy, probe))

    assert kls["kl_on"] < kls["kl_off"], (
        f"the KL term did not reduce drift: on={kls['kl_on']:.4f} off={kls['kl_off']:.4f}"
    )
