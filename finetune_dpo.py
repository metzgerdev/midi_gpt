"""Fine-tune the note models on your own edits, with Direct Preference Optimization.

Generate a run, open its MIDI in a DAW, change what you would have played differently,
and export the result back into the run folder as `stems/<role>_edited.mid`, leaving the
original `stems/<role>.mid` where it is. That pair is the training signal:

    stems/<role>.mid          REJECTED   what the model produced
    stems/<role>_edited.mid   CHOSEN     what you changed it into

Both are evaluated under identical conditioning, rebuilt from the run's metadata.json —
the same kick grid and chord chroma the original was generated with.

    python finetune_dpo.py                      # both roles, every edited run in output/
    python finetune_dpo.py --role bass --beta 0.2

Given a frozen ANCHOR and a trainable POLICY:

    r(seq)  = beta * ( logP_policy(seq) - logP_anchor(seq) )
    margin  = r(chosen) - r(rejected)
    loss    = -log sigmoid(margin) + lambda * KL(policy || anchor)

The first term raises the policy's likelihood of your edit and lowers it for what it
replaced. Beta only scales that reward, so it cannot bound drift on its own. Two things
do: the KL term, whose lambda is adapted to hold --kl-target, and a FIXED anchor —
by default the untuned base, not whichever ftN you started from, so drift is measured
from one place across rounds instead of compounding.

Chosen and rejected are both exactly NOTE_STEPS long, so the length bias that affects DPO
on variable-length text cannot arise here.

Writes the next checkpoint in the ftN sequence. make_track.py selects the highest ftN
automatically, so the next batch you edit comes from the model that absorbed this one.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import re
from pathlib import Path

import librosa
import mido
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from audio_features import onset_grid
from chords import progression_to_track
from config import (
    GRID_REF_BPM, GRID_STEPS, NOTE_BOS, NOTE_EOS, NOTE_MIDI_HI, NOTE_MIDI_LO,
    NOTE_PITCH0, NOTE_REST, NOTE_STEPS, NOTE_SUSTAIN, NOTE_VOCAB_SIZE, ROLE_CENTER,
    ROLE_MONO, SAMPLE_RATE,
)
from make_track import (
    CKPT_DIR, DRUM_DIR, OUTPUT_DIR, align_to_grid_tempo, latest_ckpt, pick_device,
)
from midi_utils import midi_to_step_grid, notes_to_tokens, octave_fit
from model import HarmonicNoteGPT, generate_notes, note_chord_cond, note_grid_cond

TRAINING_DATA = Path(__file__).parent / "training_data"


def next_ft_path(role: str) -> Path:
    """The next unused <role>_notes_gpt_ftN.pt — never overwrites an existing checkpoint."""
    n = 1
    while (CKPT_DIR / f"{role}_notes_gpt_ft{n}.pt").exists():
        n += 1
    return CKPT_DIR / f"{role}_notes_gpt_ft{n}.pt"


def find_drum_loop(name: str) -> Path | None:
    """Find a loop a past run was conditioned on in either loop directory, else None."""
    for directory in (DRUM_DIR, TRAINING_DATA / "drum_loops"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def run_conditioning(run_dir: Path):
    """metadata.json -> (grid64, [chroma per section], meta): what the run was generated under.

    Rebuilt rather than stored, so a pair is always scored against the exact conditioning
    that produced it — including the drum loop's tempo alignment.
    """
    meta = json.loads((run_dir / "metadata.json").read_text())

    drum_path = find_drum_loop(meta["drum_loop"])
    if drum_path is None:
        raise SystemExit(
            f"{run_dir.name} was conditioned on {meta['drum_loop']}, which is not in "
            f"{DRUM_DIR} or {TRAINING_DATA / 'drum_loops'}. "
            f"The kick grid cannot be rebuilt without it."
        )
    audio, _ = librosa.load(drum_path, sr=SAMPLE_RATE, mono=True)
    audio = align_to_grid_tempo(audio, meta.get("drum_bpm", GRID_REF_BPM))
    envelope = onset_grid(audio)
    binned = np.array([b.max() if b.size else 0.0
                       for b in np.array_split(envelope, GRID_STEPS)])
    grid32 = (binned > 0.25).astype(np.float32)
    grid64 = np.tile(grid32, NOTE_STEPS // len(grid32))[:NOTE_STEPS]

    # runs made with --chords color carry one progression per section
    progressions = meta.get("progressions") or [meta["progression"]] * meta["sections"]
    chromas = [progression_to_track(p, NOTE_STEPS) for p in progressions]
    return grid64, chromas, meta


def tokenize_sections(mid_path: Path, role: str, grid64, chromas):
    """MIDI -> {(section, key_shift): (x, y, grid_tok, chord_tok)}.

    No note-count filter, so chosen and rejected line up one to one for pairing.
    """
    if not mid_path.exists():
        return {}
    pitch = midi_to_step_grid(mido.MidiFile(mid_path), mono=ROLE_MONO[role])
    if pitch is None:
        return {}
    grid_tok = note_grid_cond(grid64)                       # rhythm is shift-invariant
    out = {}
    for section in range(max(1, len(pitch) // NOTE_STEPS)):
        segment = pitch[section * NOTE_STEPS:(section + 1) * NOTE_STEPS]
        if len(segment) < NOTE_STEPS:
            segment = np.concatenate([segment, np.full(NOTE_STEPS - len(segment), -1, int)])
        chroma = chromas[min(section, len(chromas) - 1)]
        for shift in range(12):                             # notes and chroma rotate together
            tokens = notes_to_tokens(octave_fit(segment, shift, ROLE_CENTER[role]))
            seq = np.concatenate([[NOTE_BOS], tokens, [NOTE_EOS]]).astype(np.int64)
            rolled = np.roll(chroma, shift, axis=1).astype(np.float32)
            out[(section, shift)] = (seq[:-1], seq[1:], grid_tok, note_chord_cond(rolled))
    return out


def preference_pairs(run_dir: Path, role: str):
    """(chosen, rejected) for every section of one run that you actually changed."""
    grid64, chromas, meta = run_conditioning(run_dir)
    chosen = tokenize_sections(run_dir / "stems" / f"{role}_edited.mid", role, grid64, chromas)
    rejected = tokenize_sections(run_dir / "stems" / f"{role}.mid", role, grid64, chromas)
    pairs, skipped = [], 0
    for key, take in chosen.items():
        leave = rejected.get(key)
        if leave is None:
            continue
        if np.array_equal(take[1], leave[1]):               # identical after tokenization
            skipped += 1
            continue
        pairs.append((take, leave))
    return pairs, skipped // 12, meta


def to_loader(pairs, batch_size, shuffle, seed=None):
    """Conditioning is shared within a pair, so it is stored once."""
    def column(side, field):
        return torch.tensor(np.array([p[side][field] for p in pairs]))
    dataset = TensorDataset(
        column(0, 0).long(), column(0, 1).long(),           # chosen   x, y
        column(1, 0).long(), column(1, 1).long(),           # rejected x, y
        column(0, 2).float(), column(0, 3).float(),         # grid, chord
    )
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def sequence_logprob(model, x, y, grid, chord):
    """Sum_t log P(y_t | y_<t, grid, chord) — the model's log-likelihood of the sequence."""
    logp = F.log_softmax(model(x, cond=None, cond_seq=grid, chord_seq=chord), dim=-1)
    return logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum(-1)


def anchor_contexts(anchor, pairs, n, device, seed=0):
    """Sequences sampled from the anchor, with its log-probabilities cached.

    KL is measured here rather than on the pairs, which DPO is deliberately moving.
    """
    torch.manual_seed(seed)
    stride = max(1, len(pairs) // max(n, 1))
    xs, gs, cs = [], [], []
    for chosen, _ in pairs[::stride][:n]:
        _, _, grid_tok, chord_tok = chosen
        out = generate_notes(anchor, grid_tok, NOTE_BOS, NOTE_EOS, max_steps=NOTE_STEPS,
                             temperature=1.0, top_p=0.98, chord_tok=chord_tok, device=device)
        tokens = out[0, 1:].cpu().numpy()
        tokens = np.pad(tokens, (0, max(0, NOTE_STEPS - len(tokens))),
                        constant_values=NOTE_REST)[:NOTE_STEPS]
        xs.append(np.concatenate([[NOTE_BOS], tokens]).astype(np.int64))
        gs.append(grid_tok)
        cs.append(chord_tok)
    x = torch.tensor(np.array(xs), device=device)
    g = torch.tensor(np.array(gs), dtype=torch.float32, device=device)
    c = torch.tensor(np.array(cs), dtype=torch.float32, device=device)
    with torch.no_grad():
        logq = F.log_softmax(anchor(x, cond=None, cond_seq=g, chord_seq=c), dim=-1)
    return x, g, c, logq


def kl_to_anchor(policy, contexts):
    """Exact mean per-token KL(policy || anchor), a full sum over the 65-token vocabulary.

    Dropout is disabled for the measurement, or its noise swamps the divergence.
    """
    x, g, c, logq = contexts
    was_training = policy.training
    policy.eval()
    try:
        logp = F.log_softmax(policy(x, cond=None, cond_seq=g, chord_seq=c), dim=-1)
    finally:
        policy.train(was_training)
    return (logp.exp() * (logp - logq)).sum(-1).mean()


def base_corpus_loader(role: str, n: int, batch_size: int, seed: int = 99):
    """A sample of the mined corpus, used only to check the base distribution survived.

    Returns None when the corpus is absent — it is not tracked in Git, and fine-tuning
    works without it. You simply lose the regression guard.
    """
    files = sorted(glob.glob(str(TRAINING_DATA / f"{role}_notes_midi" / "*.npz")))
    if not files:
        return None
    rng = np.random.default_rng(seed)
    picked = rng.choice(files, size=min(n, len(files)), replace=False)
    xs, ys, gs, cs = [], [], [], []
    for path in picked:
        d = np.load(path)
        seq = np.concatenate([[NOTE_BOS], d["tokens"], [NOTE_EOS]]).astype(np.int64)
        xs.append(seq[:-1]); ys.append(seq[1:])
        gs.append(note_grid_cond(d["grid"].astype(np.float32)))
        cs.append(note_chord_cond(d["chord"].astype(np.float32)))
    tensors = (torch.tensor(np.array(xs)), torch.tensor(np.array(ys)),
               torch.tensor(np.array(gs)), torch.tensor(np.array(cs)))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size)


def train_dpo(policy, anchor, loader, contexts, beta, lr, epochs,
              kl_target, kl_lambda, device=None):
    """Run DPO against a frozen anchor, holding KL to `kl_target`.

    Returns ``(final_kl, lambda)``; thread lambda into the next round so it does not
    restart cold. `contexts` of None disables the KL term, leaving plain DPO.
    """
    device = device or next(policy.parameters()).device
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    lam, kl = kl_lambda, 0.0
    policy.train()
    for _ in range(epochs):
        for xc, yc, xr, yr, g, c in loader:
            xc, yc, xr, yr, g, c = (t.to(device) for t in (xc, yc, xr, yr, g, c))
            policy_chosen = sequence_logprob(policy, xc, yc, g, c)
            policy_rejected = sequence_logprob(policy, xr, yr, g, c)
            with torch.no_grad():
                anchor_chosen = sequence_logprob(anchor, xc, yc, g, c)
                anchor_rejected = sequence_logprob(anchor, xr, yr, g, c)
            margin = beta * ((policy_chosen - anchor_chosen)
                             - (policy_rejected - anchor_rejected))
            loss = -F.logsigmoid(margin).mean()
            if contexts is not None:
                divergence = kl_to_anchor(policy, contexts)
                kl = float(divergence.detach())
                loss = loss + lam * divergence
            opt.zero_grad(); loss.backward(); opt.step()
            # Steer lambda toward the budget: double on a violation, ease off by a fifth.
            if contexts is not None:
                if kl > 1.5 * kl_target:
                    lam *= 2.0
                elif kl < kl_target / 1.5:
                    lam = max(lam / 1.2, 1e-2)
    policy.eval()
    return kl, lam


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("bass", "arp", "both"), default="both")
    ap.add_argument("--runs", type=Path, default=OUTPUT_DIR,
                    help="where to look for edited runs (default: output/)")
    ap.add_argument("--ref-ckpt", type=Path, default=None,
                    help="starting checkpoint (default: the highest ftN)")
    ap.add_argument("--anchor-ckpt", type=Path, default=None,
                    help="fixed checkpoint the margin and KL are measured against "
                         "(default: the untuned <role>_notes_gpt.pt). Point this at "
                         "--ref-ckpt to restore the old rolling-anchor behaviour.")
    ap.add_argument("--kl-target", type=float, default=0.05,
                    help="KL budget against the anchor; 0 disables the KL term")
    ap.add_argument("--kl-lambda", type=float, default=1.0,
                    help="starting KL weight; adapted every step to hold --kl-target")
    ap.add_argument("--anchor-samples", type=int, default=32,
                    help="sequences sampled from the anchor to measure KL on")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="reward scale on the preference margin")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", type=int, default=300,
                    help="base-corpus examples for the regression check (0 to skip)")
    ap.add_argument("--device", default=None, choices=("cpu", "mps", "cuda", "auto"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report the pairs that would be used, then stop")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    roles = ["bass", "arp"] if args.role == "both" else [args.role]

    weights = torch.ones(NOTE_VOCAB_SIZE, device=device)     # regression check only
    weights[NOTE_SUSTAIN] = 0.3
    weights[NOTE_PITCH0:NOTE_PITCH0 + (NOTE_MIDI_HI - NOTE_MIDI_LO + 1)] = 3.0

    for role in roles:
        pairs, skipped = [], 0
        for edited in sorted(args.runs.glob(f"*/stems/{role}_edited.mid")):
            run_dir = edited.parent.parent
            found, unchanged, meta = preference_pairs(run_dir, role)
            pairs += found
            skipped += unchanged
            print(f"[{role}] {run_dir.name}: {len(found) // 12} edited section(s)"
                  + (f", {unchanged} unchanged" if unchanged else ""), flush=True)

        if not pairs:
            print(f"[{role}] nothing to learn from. Export an edit as "
                  f"stems/{role}_edited.mid beside the original stems/{role}.mid, "
                  f"and change at least one note's pitch or step.")
            if skipped:
                print(f"[{role}] {skipped} section(s) were identical after tokenization — "
                      "velocity and sub-grid timing edits cannot be represented.")
            continue

        print(f"[{role}] {len(pairs) // 12} distinct sections x 12 keys = {len(pairs)} pairs",
              flush=True)
        if args.dry_run:
            continue

        ref_path = args.ref_ckpt or latest_ckpt(role)
        saved = torch.load(ref_path, map_location=device, weights_only=False)
        policy = HarmonicNoteGPT(saved["config"]).to(device)
        policy.load_state_dict(saved["model"])

        # Fixed anchor, not the starting policy: re-anchoring each round lets drift compound.
        anchor_path = args.anchor_ckpt or (CKPT_DIR / f"{role}_notes_gpt.pt")
        if anchor_path.resolve() == ref_path.resolve():
            reference = copy.deepcopy(policy).eval().requires_grad_(False)
        else:
            anchor_saved = torch.load(anchor_path, map_location=device, weights_only=False)
            reference = HarmonicNoteGPT(anchor_saved["config"]).to(device)
            reference.load_state_dict(anchor_saved["model"])
            reference.eval().requires_grad_(False)

        contexts = (anchor_contexts(reference, pairs, args.anchor_samples, device, args.seed)
                    if args.kl_target > 0 else None)

        loader = to_loader(pairs, args.batch_size, shuffle=True, seed=args.seed)
        eval_loader = to_loader(pairs, args.batch_size, shuffle=False)
        base_loader = base_corpus_loader(role, args.replay, args.batch_size) if args.replay else None
        if args.replay and base_loader is None:
            print(f"[{role}] no mined corpus in training_data/ — skipping the regression check",
                  flush=True)

        def measure():
            """Reward margin and preference accuracy, plus base-corpus loss if available."""
            policy.eval()
            margins, base, batches = [], 0.0, 0
            with torch.no_grad():
                for xc, yc, xr, yr, g, c in eval_loader:
                    xc, yc, xr, yr, g, c = (t.to(device) for t in (xc, yc, xr, yr, g, c))
                    margins.append(
                        (sequence_logprob(policy, xc, yc, g, c) - sequence_logprob(reference, xc, yc, g, c))
                        - (sequence_logprob(policy, xr, yr, g, c) - sequence_logprob(reference, xr, yr, g, c))
                    )
                if base_loader is not None:
                    for xb, yb, gb, cb in base_loader:
                        xb, yb, gb, cb = (t.to(device) for t in (xb, yb, gb, cb))
                        base += F.cross_entropy(
                            policy(xb, cond=None, cond_seq=gb, chord_seq=cb).flatten(0, 1),
                            yb.flatten(), weight=weights).item()
                        batches += 1
            policy.train()
            margins = torch.cat(margins)
            return (margins.mean().item(), (margins > 0).float().mean().item(),
                    base / batches if batches else None)

        margin0, acc0, base0 = measure()

        # Budget what this run adds, not the gap earlier rounds already opened.
        kl_start = 0.0
        if contexts is not None:
            with torch.no_grad():
                kl_start = float(kl_to_anchor(policy, contexts))

        kl, _ = train_dpo(policy, reference, loader, contexts, beta=args.beta, lr=args.lr,
                          epochs=args.epochs, kl_target=kl_start + args.kl_target,
                          kl_lambda=args.kl_lambda, device=device)
        margin1, acc1, base1 = measure()

        out_path = next_ft_path(role)
        torch.save({"model": policy.state_dict(), "config": saved["config"], "method": "dpo",
                    "finetuned_from": ref_path.name, "anchor": anchor_path.name,
                    "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
                    "seed": args.seed, "kl_target": args.kl_target, "kl_final": kl,
                    "pairs": len(pairs) // 12}, out_path)

        print(f"[{role}] reward margin {margin0:+.3f} -> {margin1:+.3f}   "
              f"preference accuracy {acc0:.0%} -> {acc1:.0%}")
        if contexts is not None:
            added = kl - kl_start
            verdict = ("ok" if added <= args.kl_target * 1.5
                       else "WARNING: this run went over the KL budget")
            print(f"[{role}] KL from {anchor_path.name} {kl_start:.4f} -> {kl:.4f}   "
                  f"this run added {added:+.4f} of {args.kl_target:.3f}   {verdict}")
        if base0 is not None:
            verdict = "ok" if base1 < base0 * 1.15 else "WARNING: base distribution regressed >15%"
            print(f"[{role}] base-corpus loss {base0:.3f} -> {base1:.3f}   {verdict}")
        print(f"[{role}] saved {out_path.name} — make_track.py will now use it by default; "
              f"--base runs the untuned models", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
