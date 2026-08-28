"""Train a note model from scratch, recording per-epoch train AND val loss.

WRITES NO CHECKPOINT. train_notes.py keeps only the best val loss, so the shipped
runs left no curve to plot; this reproduces its hyperparameters and records the
history instead. There is deliberately no torch.save call anywhere in this file,
so it cannot overwrite anything in checkpoints/.

    PYTHONPATH=. python experiments/loss-curves/train_with_history.py --role bass

--split augmented (default) reproduces train_notes.py exactly: random_split over the
12x-augmented corpus, so a phrase's transposed siblings straddle the train/val line.
--split grouped holds out whole phrases (all 12 keys follow the phrase), which is the
stricter measurement.
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.config import (NOTE_BOS, NOTE_EOS, NOTE_MIDI_HI, NOTE_MIDI_LO, NOTE_PITCH0,
                          NOTE_STEPS, NOTE_SUSTAIN, NOTE_VOCAB_SIZE)
from model.note_model import HarmonicNoteGPT, note_chord_cond, note_grid_cond
from utils.device import pick_device

NOTE_CONFIG = {"vocab_size": NOTE_VOCAB_SIZE, "context_length": NOTE_STEPS + 4,
               "emb_dim": 128, "n_heads": 4, "n_layers": 3, "drop_rate": 0.1,
               "qkv_bias": False}


def load_corpus(role: str, min_notes: int = 3):
    """Tensors plus a phrase id per row, shared by that phrase's 12 transpositions."""
    xs, ys, gs, cs, phrase = [], [], [], [], []
    for path in sorted(glob.glob(f"training_data/{role}_notes_midi/*.npz")):
        d = np.load(path)
        tokens = d["tokens"].astype(np.int64)
        if int((tokens >= NOTE_PITCH0).sum()) < min_notes:
            continue
        seq = np.concatenate([[NOTE_BOS], tokens, [NOTE_EOS]]).astype(np.int64)
        xs.append(seq[:-1]); ys.append(seq[1:])
        gs.append(note_grid_cond(d["grid"].astype(np.float32)))
        cs.append(note_chord_cond(d["chord"].astype(np.float32)))
        phrase.append(path.rsplit("_k", 1)[0])
    ids = {p: i for i, p in enumerate(sorted(set(phrase)))}
    return (torch.tensor(np.array(xs)), torch.tensor(np.array(ys)),
            torch.tensor(np.array(gs), dtype=torch.float32),
            torch.tensor(np.array(cs), dtype=torch.float32),
            np.array([ids[p] for p in phrase]))


def split_indices(phrase_ids, strategy="augmented", val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    if strategy == "augmented":
        perm = rng.permutation(len(phrase_ids))
        cut = max(1, int(val_frac * len(phrase_ids)))
        return perm[cut:], perm[:cut]
    if strategy == "grouped":
        uniq = np.unique(phrase_ids)
        perm = rng.permutation(len(uniq)); cut = max(1, int(val_frac * len(uniq)))
        val = set(uniq[perm[:cut]].tolist())
        mask = np.array([p in val for p in phrase_ids])
        return np.where(~mask)[0], np.where(mask)[0]
    raise ValueError(strategy)


def train_with_history(role="bass", strategy="augmented", epochs=150, batch_size=16,
                       lr=5e-4, weight_decay=0.05, seed=0, device=None, log_every=10):
    device = device or pick_device("auto")
    X, Y, G, C, phrase_ids = load_corpus(role)
    tr, va = split_indices(phrase_ids, strategy, seed=seed)
    torch.manual_seed(seed)
    mk = lambda idx, sh: DataLoader(TensorDataset(X[idx], Y[idx], G[idx], C[idx]),
                                    batch_size=batch_size, shuffle=sh, drop_last=sh)
    train_dl, val_dl = mk(tr, True), mk(va, False)

    weights = torch.ones(NOTE_VOCAB_SIZE, device=device)
    weights[NOTE_SUSTAIN] = 0.3
    weights[NOTE_PITCH0:NOTE_PITCH0 + (NOTE_MIDI_HI - NOTE_MIDI_LO + 1)] = 3.0
    model = HarmonicNoteGPT(NOTE_CONFIG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def evaluate(dl):
        model.eval(); tot = n = 0
        with torch.no_grad():
            for xb, yb, gb, cb in dl:
                xb, yb, gb, cb = (t.to(device) for t in (xb, yb, gb, cb))
                tot += F.cross_entropy(
                    model(xb, cond=None, cond_seq=gb, chord_seq=cb).flatten(0, 1),
                    yb.flatten(), weight=weights).item(); n += 1
        return tot / max(n, 1)

    hist = {"role": role, "strategy": strategy, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "weight_decay": weight_decay, "seed": seed,
            "device": str(device), "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_train_phrases": int(len(set(phrase_ids[tr].tolist()))),
            "n_val_phrases": int(len(set(phrase_ids[va].tolist()))),
            "shared_phrases": int(len(set(phrase_ids[tr].tolist())
                                      & set(phrase_ids[va].tolist()))),
            "train": [], "val": []}
    start = time.time()
    for ep in range(epochs):
        model.train(); run = n = 0
        for xb, yb, gb, cb in train_dl:
            xb, yb, gb, cb = (t.to(device) for t in (xb, yb, gb, cb))
            loss = F.cross_entropy(
                model(xb, cond=None, cond_seq=gb, chord_seq=cb).flatten(0, 1),
                yb.flatten(), weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); n += 1
        hist["train"].append(run / max(n, 1))
        hist["val"].append(evaluate(val_dl))
        if ep == 0 or (ep + 1) % log_every == 0:
            print(f"  epoch {ep+1:>3}/{epochs}  train {hist['train'][-1]:.4f}  "
                  f"val {hist['val'][-1]:.4f}  ({time.time()-start:.0f}s)", flush=True)
    hist["seconds"] = round(time.time() - start, 1)
    hist["best_val"] = min(hist["val"])
    hist["best_epoch"] = int(np.argmin(hist["val"])) + 1
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="bass", choices=("bass", "arp"))
    ap.add_argument("--split", default="augmented", choices=("augmented", "grouped"))
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    h = train_with_history(a.role, a.split, epochs=a.epochs)
    print(f"\ndone in {h['seconds']}s   best val {h['best_val']:.4f} @ epoch {h['best_epoch']}")
    print(f"{h['n_train']} train / {h['n_val']} val rows   "
          f"({h['n_train_phrases']} / {h['n_val_phrases']} phrases, "
          f"{h['shared_phrases']} phrases on both sides)")
    out = Path(f"experiments/loss-curves/history_{a.role}_{a.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(h, open(out, "w"), indent=1)
    print(f"-> {out}")
