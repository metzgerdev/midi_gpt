"""Train a note model from scratch on the mined corpus.

ARCHIVAL. Requires a mined corpus, which this repo does not ship, so it exits immediately
on a fresh clone. Building one needs `mine_corpus`, which is archival for the same reason.
To adapt the models, fine-tune the shipped checkpoints with `python -m train.finetune_dpo`
instead.

There is no pretraining stage. The model is trained directly on the target distribution:
next-token cross-entropy over one token per sixteenth note, conditioned at every step on
the kick grid and the chord chroma stored alongside each example.

    python -m train.train_notes --role bass
    python -m train.train_notes --role arp --epochs 200

Training is teacher-forced — a single forward pass covers all 65 positions at once, with
a causal mask preventing lookahead — so an epoch is fast even on a laptop CPU.

The loss is class-weighted. A bassline is mostly sustain and rest, and unweighted
cross-entropy learns to predict those and emit almost no notes. Upweighting pitch
onsets and downweighting sustain is what makes the model play.

Writes checkpoints/<role>_notes_gpt.pt, keeping the best validation loss. That is the
base checkpoint the fine-tuning scripts start from; --base in make_track.py selects it.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from utils.config import (
    CKPT_DIR, NOTE_BOS, NOTE_EOS, NOTE_MIDI_HI, NOTE_MIDI_LO, NOTE_PITCH0,
    NOTE_STEPS, NOTE_SUSTAIN, NOTE_VOCAB_SIZE,
    TRAINING_DATA,
)
from utils.device import pick_device
from model.note_model import HarmonicNoteGPT, note_chord_cond, note_grid_cond


# context_length allows BOS + NOTE_STEPS + EOS with room to spare
NOTE_CONFIG = {"vocab_size": NOTE_VOCAB_SIZE, "context_length": NOTE_STEPS + 4,
               "emb_dim": 128, "n_heads": 4, "n_layers": 3, "drop_rate": 0.1,
               "qkv_bias": False}


DIGEST_RE = re.compile(r"^.+_([0-9a-f]{10})__\d+_k\d+$")


def phrase_families(data_dir: Path, names: list[str]) -> list[str]:
    """The phrase each example belongs to, read from the mining manifest.

    Falls back to the source digest when no manifest is present, which is weaker but
    still better than nothing: it groups the twelve keys this repo added, though not
    the twelve the source corpus had already added before mining.
    """
    role = data_dir.name.replace("_notes_midi", "")
    manifest_path = TRAINING_DATA / "manifests" / f"{role}.json"
    family_of = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        family_of = {s["digest"]: s["family"] for s in manifest["sources"]}
    out = []
    for name in names:
        matched = DIGEST_RE.match(Path(name).stem)
        digest = matched[1] if matched else Path(name).stem
        out.append(family_of.get(digest, digest))
    return out


def grouped_split(families: list[str], val_split: float, seed: int):
    """Indices for a split no phrase straddles.

    Shuffles whole phrases, not rows, and fills validation until it reaches the target
    share of examples. A random split over this corpus leaves *zero* clean validation
    examples — every one has another key of its own source file in train, and ~half have
    a byte-identical twin — so the gap it reports is a train-train gap.
    """
    by_family: dict[str, list[int]] = {}
    for i, family in enumerate(families):
        by_family.setdefault(family, []).append(i)

    order = sorted(by_family)                       # deterministic before shuffling
    random.Random(seed).shuffle(order)
    target = max(1, int(val_split * len(families)))

    val_idx: list[int] = []
    for family in order:
        if len(val_idx) >= target:
            break
        val_idx += by_family[family]
    held = set(val_idx)
    train_idx = [i for i in range(len(families)) if i not in held]
    if not train_idx or not val_idx:
        raise SystemExit(
            f"a grouped split of {len(by_family)} phrases at val_split={val_split} "
            f"left one side empty; lower --val-split or use --split-by random"
        )
    return sorted(train_idx), sorted(val_idx)


class NoteCorpus(Dataset):
    """The mined .npz set: tokens plus their two aligned conditioning tracks."""

    def __init__(self, data_dir: Path, min_notes: int = 3):
        files = sorted(glob.glob(str(data_dir / "*.npz")))
        if not files:
            raise SystemExit(
                f"no .npz in {data_dir}. Build the corpus first:\n"
                f"    python -m train.mine_corpus --role <role>"
            )
        xs, ys, grids, chords, names = [], [], [], [], []
        for path in files:
            example = np.load(path)
            tokens = example["tokens"].astype(np.int64)
            if int((tokens >= NOTE_PITCH0).sum()) < min_notes:
                continue
            sequence = np.concatenate([[NOTE_BOS], tokens, [NOTE_EOS]]).astype(np.int64)
            xs.append(sequence[:-1])                    # input
            ys.append(sequence[1:])                     # target: the next token
            grids.append(note_grid_cond(example["grid"].astype(np.float32)))
            chords.append(note_chord_cond(example["chord"].astype(np.float32)))
            names.append(Path(path).name)
        self.x = torch.tensor(np.array(xs))
        self.y = torch.tensor(np.array(ys))
        self.grid = torch.tensor(np.array(grids), dtype=torch.float32)
        self.chord = torch.tensor(np.array(chords), dtype=torch.float32)
        # Kept so a split can group on the phrase behind each row rather than the row.
        self.names = names
        self.families = phrase_families(data_dir, names)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.grid[i], self.chord[i]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="bass", choices=("bass", "arp"))
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="default: training_data/<role>_notes_midi")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-notes", type=int, default=3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--split-by", default="family", choices=("family", "random"),
                    help="'family' keeps every key of a phrase on one side (default); "
                         "'random' reproduces the leaking split the shipped "
                         "checkpoints were trained under")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: checkpoints/<role>_notes_gpt.pt")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out if it already exists")
    ap.add_argument("--device", default=None, choices=("cpu", "mps", "cuda", "auto"))
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    data_dir = args.data_dir or TRAINING_DATA / f"{args.role}_notes_midi"
    out = args.out or CKPT_DIR / f"{args.role}_notes_gpt.pt"

    # Checked before the corpus loads, so a refusal costs a second rather than an epoch.
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. Training from scratch would replace the checkpoint "
            f"every other script loads by default.\n"
            f"    --out <path>   write somewhere else\n"
            f"    --force        overwrite it"
        )

    dataset = NoteCorpus(data_dir, min_notes=args.min_notes)
    if args.split_by == "family":
        train_idx, val_idx = grouped_split(dataset.families, args.val_split,
                                           args.split_seed)
        train_set, val_set = Subset(dataset, train_idx), Subset(dataset, val_idx)
        held = len({dataset.families[i] for i in val_idx})
        split_note = (f"grouped by phrase ({held} of "
                      f"{len(set(dataset.families))} held out)")
    else:
        n_val = max(1, int(args.val_split * len(dataset)))
        train_set, val_set = random_split(
            dataset, [len(dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.split_seed))
        split_note = "RANDOM — phrases straddle the split, so val loss is not held out"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = HarmonicNoteGPT(NOTE_CONFIG).to(device)
    print(f"device {device}   examples {len(dataset)} "
          f"(train {len(train_set)} / val {len(val_set)})   "
          f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"split:  {split_note}", flush=True)

    # Upweight onsets, downweight sustain: without this the model predicts the majority
    # classes and emits a sparse drone.
    weights = torch.ones(NOTE_VOCAB_SIZE, device=device)
    weights[NOTE_SUSTAIN] = 0.3
    weights[NOTE_PITCH0:NOTE_PITCH0 + (NOTE_MIDI_HI - NOTE_MIDI_LO + 1)] = 3.0

    def loss_of(batch):
        x, y, grid, chord = (t.to(device) for t in batch)
        logits = model(x, cond_seq=grid, chord_seq=chord)
        return F.cross_entropy(logits.flatten(0, 1), y.flatten(), weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Best-so-far goes to a sidecar and is moved into place only once training finishes,
    # so an interrupted run leaves the existing checkpoint untouched rather than a
    # half-trained one in its place. The sidecar keeps the crash resilience of saving
    # every improvement.
    partial = out.with_name(out.name + ".partial")
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss_of(batch).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_of(b).item() for b in val_loader]))
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "config": NOTE_CONFIG, "best": best},
                       partial)
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch + 1}/{args.epochs}   val {val:.3f}   best {best:.3f}",
                  flush=True)

    if not partial.exists():
        raise SystemExit(f"no epoch completed, so nothing was written to {out}")
    partial.replace(out)
    print(f"done. best val {best:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
