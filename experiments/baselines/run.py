"""Reference points for chord_fit and kick_lock. Is 0.814 good? This answers it.

The repo reports chord_fit 0.814 and kick_lock 0.912 with clustered standard errors, and
those numbers mean nothing on their own — nobody can say whether 0.814 is good, including
the person who measured it. A metric without a floor and a ceiling is a number, not a
result. This scores five systems on identical conditioning:

  random        legal tokens, uniform. The floor: what a metric gives you for free.
  markov        bigram over the corpus. Beats random on anything that rewards local
                plausibility, so it separates "learned the note distribution" from
                "learned the conditioning".
  unconditioned the model on a zeroed grid and zeroed chroma. The ablation, and the one
                that makes the architecture claim falsifiable — if this scores as well as
                the conditioned model, the conditioning does nothing.
  model         the shipped checkpoints, best-of-N exactly as make_track runs them.
  corpus        human-authored training MIDI scored on its own metrics. The ceiling —
                and if the model beats it, the metric is gameable and that is the finding.

    PYTHONPATH=. python experiments/baselines/run.py             # -> baselines.json

Uses `clustered_se` for the interval: each of the 12 transpositions of a phrase is the
same music, so treating them as independent understates the error by about sqrt(12).

Inference only. Nothing here trains or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from model.checkpoints import latest_ckpt, load_note_model
from model.note_model import generate_notes, note_chord_cond, note_grid_cond
from utils.config import (
    NOTE_BOS, NOTE_EOS, NOTE_PITCH0, NOTE_REST, NOTE_STEPS, NOTE_SUSTAIN,
    NOTE_VOCAB, NOTE_VOCAB_SIZE, TRAINING_DATA,
)
from utils.metric_utils import clustered_se
from utils.scoring import fit_and_lock

HERE = Path(__file__).resolve().parent


def corpus_examples(role: str, limit: int | None = None):
    """Every mined example with its conditioning, keyed by phrase for clustering."""
    manifest_path = TRAINING_DATA / "manifests" / f"{role}.json"
    family_of = {}
    if manifest_path.exists():
        family_of = {s["digest"]: s["family"]
                     for s in json.loads(manifest_path.read_text())["sources"]}

    import re
    digest_re = re.compile(r"^.+_([0-9a-f]{10})__\d+_k\d+$")
    rows = []
    for path in sorted((TRAINING_DATA / f"{role}_notes_midi").glob("*.npz")):
        example = np.load(path)
        matched = digest_re.match(path.stem)
        digest = matched[1] if matched else path.stem
        rows.append({
            "tokens": example["tokens"].astype(np.int64),
            "grid": example["grid"].astype(np.float32),
            "chord": example["chord"].astype(np.float32),
            "family": family_of.get(digest, digest),
        })
        if limit and len(rows) >= limit:
            break
    return rows


def legal_random(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform over tokens that are legal at each position.

    Held to the same legality rule the sampler enforces — SUSTAIN cannot open a sequence
    or follow a rest — so the floor is "random *valid* music", not "random noise that
    the metric rejects on a technicality".
    """
    tokens, previous = [], NOTE_BOS
    for _ in range(n):
        choices = [t for t in range(NOTE_VOCAB)
                   if not (t == NOTE_SUSTAIN and previous in (NOTE_BOS, NOTE_REST))]
        previous = int(rng.choice(choices))
        tokens.append(previous)
    return np.array(tokens, dtype=np.int64)


def fit_bigram(rows) -> np.ndarray:
    """Transition counts over the corpus, Laplace-smoothed."""
    counts = np.ones((NOTE_VOCAB_SIZE, NOTE_VOCAB_SIZE), dtype=np.float64)
    for row in rows:
        sequence = np.concatenate([[NOTE_BOS], row["tokens"]])
        for a, b in zip(sequence[:-1], sequence[1:]):
            counts[a, b] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def sample_bigram(probabilities, rng, n) -> np.ndarray:
    tokens, previous = [], NOTE_BOS
    for _ in range(n):
        p = probabilities[previous].copy()
        p[NOTE_BOS] = p[NOTE_EOS] = 0.0
        if previous in (NOTE_BOS, NOTE_REST):
            p[NOTE_SUSTAIN] = 0.0
        p = p / p.sum()
        previous = int(rng.choice(len(p), p=p))
        tokens.append(previous)
    return np.array(tokens, dtype=np.int64)


def sample_model(model, grid, chord, device, temperature, seed) -> np.ndarray:
    torch.manual_seed(seed)
    out = generate_notes(model, note_grid_cond(grid), NOTE_BOS, NOTE_EOS,
                         max_steps=NOTE_STEPS, temperature=temperature, top_p=0.98,
                         chord_tok=note_chord_cond(chord), device=device)
    tokens = out[0, 1:].cpu().numpy()
    return np.pad(tokens, (0, max(0, NOTE_STEPS - len(tokens))),
                  constant_values=NOTE_REST)[:NOTE_STEPS]


def best_of(sampler, chord, grid, n_cand):
    """make_track's selection rule, so the model row is what a user actually hears."""
    best, best_score, best_scores = None, -1e9, (0.0, 0.0)
    for c in range(n_cand):
        tokens = sampler(c)
        fit, lock = fit_and_lock(tokens, chord, grid)
        onsets = int((tokens >= NOTE_PITCH0).sum())
        density = 1.0 - min(abs(onsets - 10) / 10.0, 1.0)
        score = 2.0 * fit + 1.0 * lock + 0.7 * density + (-2.0 if onsets < 3 else 0.0)
        if score > best_score:
            best, best_score, best_scores = tokens, score, (fit, lock)
    return best, best_scores


def summarise(name, scored) -> dict:
    """Cluster by phrase — the 12 keys of one phrase are not 12 independent samples."""
    families = np.array([row["family"] for row in scored])
    fits = np.array([row["fit"] for row in scored], dtype=np.float64)
    locks = np.array([row["lock"] for row in scored], dtype=np.float64)
    densities = np.array([row["density"] for row in scored], dtype=np.float64)
    return {
        "system": name,
        "n": len(scored),
        "clusters": int(len(np.unique(families))),
        "chord_fit": round(float(fits.mean()), 3),
        "chord_fit_se": round(clustered_se(fits, families), 3),
        "kick_lock": round(float(locks.mean()), 3),
        "kick_lock_se": round(clustered_se(locks, families), 3),
        "note_density": round(float(densities.mean()), 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", nargs="*", default=["bass", "arp"])
    ap.add_argument("--samples", type=int, default=240,
                    help="conditioning cases drawn from the corpus per system")
    ap.add_argument("--candidates", type=int, default=10,
                    help="best-of-N, matching make_track's default")
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=HERE / "baselines.json")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    results = {}

    for role in args.roles:
        rows = corpus_examples(role)
        rng = np.random.default_rng(args.seed)
        picked = [rows[i] for i in rng.choice(len(rows), min(args.samples, len(rows)),
                                              replace=False)]
        bigram = fit_bigram(rows)
        model = load_note_model(latest_ckpt(role), device)
        zero_grid = np.zeros(NOTE_STEPS, dtype=np.float32)
        zero_chord = np.zeros((NOTE_STEPS, 12), dtype=np.float32)

        systems = {name: [] for name in
                   ("random", "markov", "unconditioned", "model", "corpus")}

        for i, row in enumerate(picked):
            chord, grid, family = row["chord"], row["grid"], row["family"]

            def record(name, tokens, scores=None):
                fit, lock = scores or fit_and_lock(tokens, chord, grid)
                systems[name].append({
                    "family": family, "fit": fit, "lock": lock,
                    "density": int((tokens >= NOTE_PITCH0).sum()),
                })

            tokens, scores = best_of(
                lambda c, r=np.random.default_rng(args.seed + i): legal_random(r, NOTE_STEPS),
                chord, grid, args.candidates)
            record("random", tokens, scores)

            tokens, scores = best_of(
                lambda c, r=np.random.default_rng(args.seed + i): sample_bigram(bigram, r, NOTE_STEPS),
                chord, grid, args.candidates)
            record("markov", tokens, scores)

            # The ablation scores against the REAL conditioning while being GIVEN none:
            # that is the question — does reading the grid and chroma help hit them?
            tokens, scores = best_of(
                lambda c, i=i: sample_model(model, zero_grid, zero_chord, device,
                                            args.temperature, args.seed + i * 1000 + c),
                chord, grid, args.candidates)
            record("unconditioned", tokens, scores)

            tokens, scores = best_of(
                lambda c, i=i: sample_model(model, grid, chord, device,
                                            args.temperature, args.seed + i * 1000 + c),
                chord, grid, args.candidates)
            record("model", tokens, scores)

            record("corpus", row["tokens"])

            if (i + 1) % 40 == 0:
                print(f"  {role}: {i + 1}/{len(picked)}", flush=True)

        results[role] = [summarise(name, scored) for name, scored in systems.items()]

        print(f"\n=== {role} (n={len(picked)} conditioning cases, "
              f"best-of-{args.candidates}) ===")
        header = f"{'system':<16}{'chord_fit':>18}{'kick_lock':>18}{'density':>10}"
        print(header)
        for entry in results[role]:
            print(f"{entry['system']:<16}"
                  f"{entry['chord_fit']:>11.3f} ±{entry['chord_fit_se']:<6.3f}"
                  f"{entry['kick_lock']:>11.3f} ±{entry['kick_lock_se']:<6.3f}"
                  f"{entry['note_density']:>10.1f}")

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
