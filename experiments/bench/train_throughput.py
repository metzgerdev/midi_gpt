"""Time the training loop without training a model. Answers "how long does it take?"

The post says "about seven and a half minutes on an M-series GPU", the README says "runs
on the CPU in minutes". Neither had a measurement behind it and they do not agree about
the device, which is the part a reader would act on.

This runs the real loop — same data, same batch size, same optimizer, same class weights
— for a handful of epochs on each device, and reports the measured per-epoch cost plus
the projected cost of a full run. **It never saves a checkpoint.** The model it builds is
discarded, so this cannot disturb the shipped weights, which is the whole reason it
measures a few epochs and extrapolates instead of running the real thing to completion.

    PYTHONPATH=. python experiments/bench/train_throughput.py         # -> train_throughput.json
    PYTHONPATH=. python experiments/bench/train_throughput.py --epochs 10

Extrapolation is honest here because the loop is fixed-cost per epoch: the dataset is
fully materialised in memory by NoteCorpus, every example is the same 65 tokens, and
there is no scheduler or early stop. The reported projection is epochs x median epoch,
and the median is reported alongside so the arithmetic is checkable.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model.note_model import HarmonicNoteGPT
from train.train_notes import NOTE_CONFIG, NoteCorpus, grouped_split
from utils.config import (
    NOTE_MIDI_HI, NOTE_MIDI_LO, NOTE_PITCH0, NOTE_SUSTAIN, NOTE_VOCAB_SIZE,
    TRAINING_DATA,
)

HERE = Path(__file__).resolve().parent
FULL_RUN_EPOCHS = 150                      # train_notes.py's default


def available_devices(requested):
    if requested:
        return requested
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def time_epochs(dataset, device_name, epochs, batch_size, lr, val_split, seed):
    """Median seconds per epoch, including the validation pass."""
    device = torch.device(device_name)
    train_idx, val_idx = grouped_split(dataset.families, val_split, seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    model = HarmonicNoteGPT(NOTE_CONFIG).to(device)
    weights = torch.ones(NOTE_VOCAB_SIZE, device=device)
    weights[NOTE_SUSTAIN] = 0.3
    weights[NOTE_PITCH0:NOTE_PITCH0 + (NOTE_MIDI_HI - NOTE_MIDI_LO + 1)] = 3.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    def loss_of(batch):
        x, y, grid, chord = (t.to(device) for t in batch)
        logits = model(x, cond_seq=grid, chord_seq=chord)
        return F.cross_entropy(logits.flatten(0, 1), y.flatten(), weight=weights)

    samples = []
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss_of(batch).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_of(b).item() for b in val_loader]))
        if device_name == "mps":
            torch.mps.synchronize()
        elif device_name == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        samples.append(elapsed)
        print(f"    epoch {epoch + 1}/{epochs}  {elapsed:6.2f}s  val {val:.3f}", flush=True)

    # the first epoch pays allocator and kernel-cache costs the rest do not
    steady = samples[1:] or samples
    return samples, steady


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", nargs="*", default=["bass"])
    ap.add_argument("--devices", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=6, help="timed epochs per device")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "train_throughput.json")
    args = ap.parse_args(argv)

    results = {
        "machine": {"platform": platform.platform(), "torch": torch.__version__},
        "settings": {"epochs_timed": args.epochs, "batch_size": args.batch_size,
                     "full_run_epochs": FULL_RUN_EPOCHS, "saves_checkpoint": False},
        "roles": {},
    }

    for role in args.roles:
        dataset = NoteCorpus(TRAINING_DATA / f"{role}_notes_midi")
        print(f"=== {role}: {len(dataset)} examples ===", flush=True)
        per_device = {}
        for device_name in available_devices(args.devices):
            print(f"  {device_name}:", flush=True)
            samples, steady = time_epochs(dataset, device_name, args.epochs,
                                          args.batch_size, args.lr, args.val_split,
                                          args.seed)
            median = statistics.median(steady)
            per_device[device_name] = {
                "epochs_timed": len(samples),
                "first_epoch_s": round(samples[0], 2),
                "median_epoch_s": round(median, 2),
                "projected_full_run_s": round(median * FULL_RUN_EPOCHS, 1),
                "projected_full_run_min": round(median * FULL_RUN_EPOCHS / 60, 1),
            }
            print(f"    median {median:.2f}s/epoch -> "
                  f"{median * FULL_RUN_EPOCHS / 60:.1f} min for "
                  f"{FULL_RUN_EPOCHS} epochs", flush=True)
        results["roles"][role] = {"examples": len(dataset), "devices": per_device}

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
