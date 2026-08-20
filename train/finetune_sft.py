"""Supervised fine-tuning on your edits: pull the model toward what you kept.

ARCHIVAL. This runs without the mined corpus, but the corpus is its only anchor: without
it the replay mix and the regression check both fall away silently, and the edits are free
to overwrite the base distribution. `python -m train.finetune_dpo` is the supported path —
its KL guard is measured against a checkpoint, so it needs nothing this repo does not ship.

The lighter of the two fine-tuning levers. SFT trains on the edited MIDI alone, so it
learns what you wanted; DPO (finetune_dpo.py) also uses the original as a negative, so it
learns what you rejected. SFT needs only an edit, not a pair, which makes it the one to
reach for when you have changed a clip beyond recognition.

    python -m train.finetune_sft                    # both roles, every edited run in output/
    python -m train.finetune_sft --role bass --epochs 12

Same convention as DPO: export the edit into the run folder as stems/<role>_edited.mid.
The conditioning is rebuilt from the run's metadata.json.

Edits are mixed with a replayed sample of the mined corpus and trained at a low learning
rate, so a handful of clips cannot overwrite the base distribution. The run reports the
loss on that corpus before and after; if it climbs sharply, the edits have taken over.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.config import (
    CKPT_DIR, NOTE_MIDI_HI, NOTE_MIDI_LO, NOTE_PITCH0, NOTE_SUSTAIN, NOTE_VOCAB_SIZE,
    OUTPUT_DIR,
)
from model.checkpoints import latest_ckpt
from utils.device import pick_device
from train.finetune_dpo import (
    base_corpus_loader, next_ft_path, run_conditioning, tokenize_sections,
)
from model.note_model import HarmonicNoteGPT


def edit_examples(run_dir, role: str, min_notes: int = 3):
    """Every 4-bar section of an edited clip, across twelve keys."""
    grid64, chromas, _ = run_conditioning(run_dir)
    sections = tokenize_sections(run_dir / "stems" / f"{role}_edited.mid",
                                 role, grid64, chromas)
    return [v for v in sections.values()
            if int((v[1] >= NOTE_PITCH0).sum()) >= min_notes]


def to_loader(examples, batch_size, shuffle):
    columns = [torch.tensor(np.array([e[i] for e in examples])) for i in range(4)]
    dataset = TensorDataset(columns[0].long(), columns[1].long(),
                            columns[2].float(), columns[3].float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("bass", "arp", "both"), default="both")
    ap.add_argument("--runs", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--ref-ckpt", type=Path, default=None,
                    help="starting checkpoint (default: the highest ftN)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--edit-repeat", type=int, default=4,
                    help="how many times each edit is repeated against the replay sample")
    ap.add_argument("--replay", type=int, default=500,
                    help="mined-corpus examples mixed in to anchor the base distribution")
    ap.add_argument("--device", default=None, choices=("cpu", "mps", "cuda", "auto"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    roles = ["bass", "arp"] if args.role == "both" else [args.role]

    weights = torch.ones(NOTE_VOCAB_SIZE, device=device)
    weights[NOTE_SUSTAIN] = 0.3
    weights[NOTE_PITCH0:NOTE_PITCH0 + (NOTE_MIDI_HI - NOTE_MIDI_LO + 1)] = 3.0

    for role in roles:
        edits = []
        for edited in sorted(args.runs.glob(f"*/stems/{role}_edited.mid")):
            run_dir = edited.parent.parent
            found = edit_examples(run_dir, role)
            edits += found
            print(f"[{role}] {run_dir.name}: {len(found) // 12} section(s)", flush=True)
        if not edits:
            print(f"[{role}] no edits found. Export one as stems/{role}_edited.mid "
                  f"inside a run folder under {args.runs}.")
            continue
        print(f"[{role}] {len(edits) // 12} distinct sections x 12 keys = {len(edits)}",
              flush=True)
        if args.dry_run:
            continue

        replay = base_corpus_loader(role, args.replay, args.batch_size)
        if replay is None:
            print(f"[{role}] no mined corpus in training_data/ — training on the edits "
                  "alone, which will drift much further from the base distribution",
                  flush=True)

        ref_path = args.ref_ckpt or latest_ckpt(role)
        saved = torch.load(ref_path, map_location=device, weights_only=False)
        model = HarmonicNoteGPT(saved["config"]).to(device)
        model.load_state_dict(saved["model"])

        replay_examples_list = []
        if replay is not None:
            for xb, yb, gb, cb in replay:
                for i in range(len(xb)):
                    replay_examples_list.append((xb[i].numpy(), yb[i].numpy(),
                                                 gb[i].numpy(), cb[i].numpy()))
        loader = to_loader(edits * args.edit_repeat + replay_examples_list,
                           args.batch_size, shuffle=True)
        edit_loader = to_loader(edits, args.batch_size, shuffle=False)

        def mean_loss(dl):
            model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for x, y, g, c in dl:
                    x, y, g, c = (t.to(device) for t in (x, y, g, c))
                    total += F.cross_entropy(
                        model(x, cond=None, cond_seq=g, chord_seq=c).flatten(0, 1),
                        y.flatten(), weight=weights).item()
                    n += 1
            model.train()
            return total / max(n, 1)

        edit0 = mean_loss(edit_loader)
        base0 = mean_loss(replay) if replay is not None else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        model.train()
        for _ in range(args.epochs):
            for x, y, g, c in loader:
                x, y, g, c = (t.to(device) for t in (x, y, g, c))
                loss = F.cross_entropy(
                    model(x, cond=None, cond_seq=g, chord_seq=c).flatten(0, 1),
                    y.flatten(), weight=weights)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        edit1 = mean_loss(edit_loader)
        base1 = mean_loss(replay) if replay is not None else None

        out_path = next_ft_path(role)
        torch.save({"model": model.state_dict(), "config": saved["config"],
                    "method": "sft", "finetuned_from": ref_path.name,
                    "edits": len(edits) // 12}, out_path)

        print(f"[{role}] edit loss {edit0:.3f} -> {edit1:.3f}")
        if base0 is not None:
            verdict = "ok" if base1 < base0 * 1.15 else "WARNING: base regressed >15%"
            print(f"[{role}] base-corpus loss {base0:.3f} -> {base1:.3f}   {verdict}")
        print(f"[{role}] saved {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
