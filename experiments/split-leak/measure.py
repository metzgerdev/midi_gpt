"""How much of the validation set is already in train? Counted, not retrained.

`train_notes.py` splits with `random_split(..., manual_seed(0))` over the *augmented*
corpus, and the README reports a +0.022 generalisation gap from it. That gap is not
measuring generalisation, for two reasons that compound:

  1. Every phrase is stored in twelve keys, and a random split scatters those twelve
     across both sides. Known, and admitted in the README.
  2. The corpus was mined from a source that was *already* transposition-augmented, so
     the twelve keys are really ~144 views of 45 phrases — and a third of the examples
     are byte-identical to another example. Not known until the corpus was audited.

Both are countable against the shipped data without training anything, which is the
point: the honest number here costs a script, not a retrain.

    PYTHONPATH=. python experiments/split-leak/measure.py        # -> leak.json

Reproduces the split exactly — same seed, same ordering, same min_notes filter — by
importing NoteCorpus rather than reimplementing it, so it cannot drift from the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from train.train_notes import NoteCorpus
from utils.config import TRAINING_DATA

HERE = Path(__file__).resolve().parent
DIGEST_RE = re.compile(r"^.+_([0-9a-f]{10})__\d+_k(\d+)$")


def example_keys(data_dir: Path, min_notes: int):
    """Per-example identity, in the same order NoteCorpus builds its rows.

    NoteCorpus globs sorted and skips examples under min_notes, so walking the same glob
    with the same filter reproduces its indexing. Asserted against len(dataset) by the
    caller rather than trusted.
    """
    from utils.config import NOTE_PITCH0

    rows = []
    for path in sorted(data_dir.glob("*.npz")):
        example = np.load(path)
        tokens = example["tokens"].astype(np.int64)
        if int((tokens >= NOTE_PITCH0).sum()) < min_notes:
            continue
        content = hashlib.md5(
            tokens.tobytes() + example["grid"].tobytes() + example["chord"].tobytes()
        ).hexdigest()
        matched = DIGEST_RE.match(path.stem)
        rows.append({
            "name": path.name,
            "content": content,
            "digest": matched[1] if matched else path.stem,
            "shift": int(matched[2]) if matched else -1,
        })
    return rows


def measure(role: str, val_split: float, min_notes: int, seed: int) -> dict:
    data_dir = TRAINING_DATA / f"{role}_notes_midi"
    dataset = NoteCorpus(data_dir, min_notes=min_notes)
    rows = example_keys(data_dir, min_notes)
    assert len(rows) == len(dataset), (
        f"{len(rows)} keys vs {len(dataset)} dataset rows — the orderings have diverged"
    )

    manifest_path = TRAINING_DATA / "manifests" / f"{role}.json"
    family_of = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        family_of = {s["digest"]: s["family"] for s in manifest["sources"]}

    # the split train_notes.py actually performs
    n_val = max(1, int(val_split * len(dataset)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_rows = [rows[i] for i in train_set.indices]
    val_rows = [rows[i] for i in val_set.indices]

    train_content = {r["content"] for r in train_rows}
    train_digest = {r["digest"] for r in train_rows}
    train_family = {family_of.get(r["digest"], r["digest"]) for r in train_rows}

    identical = sum(r["content"] in train_content for r in val_rows)
    same_source = sum(r["digest"] in train_digest for r in val_rows)
    same_family = sum(family_of.get(r["digest"], r["digest"]) in train_family
                      for r in val_rows)
    clean = sum(
        r["content"] not in train_content
        and family_of.get(r["digest"], r["digest"]) not in train_family
        for r in val_rows
    )

    return {
        "role": role,
        "examples": len(dataset),
        "train": len(train_rows),
        "val": len(val_rows),
        "val_split": val_split,
        "seed": seed,
        "leak": {
            "byte_identical_twin_in_train": identical,
            "same_source_file_in_train": same_source,
            "same_phrase_family_in_train": same_family,
            "clean": clean,
        },
        "leak_pct": {
            "byte_identical_twin_in_train": round(100 * identical / len(val_rows), 1),
            "same_source_file_in_train": round(100 * same_source / len(val_rows), 1),
            "same_phrase_family_in_train": round(100 * same_family / len(val_rows), 1),
            "clean": round(100 * clean / len(val_rows), 1),
        },
        "corpus": {
            "distinct_content": len({r["content"] for r in rows}),
            "duplicate_examples": len(rows) - len({r["content"] for r in rows}),
            "source_files": len({r["digest"] for r in rows}),
            "phrase_families": len({family_of.get(r["digest"], r["digest"]) for r in rows}),
        },
        "families_held_out_entirely": sorted(
            {family_of.get(r["digest"], r["digest"]) for r in val_rows} - train_family
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", nargs="*", default=["bass", "arp"])
    ap.add_argument("--val-split", type=float, default=0.15,
                    help="train_notes.py's default")
    ap.add_argument("--min-notes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="train_notes.py's split seed")
    ap.add_argument("--out", type=Path, default=HERE / "leak.json")
    args = ap.parse_args(argv)

    results = {}
    for role in args.roles:
        result = measure(role, args.val_split, args.min_notes, args.seed)
        results[role] = result
        leak, pct = result["leak"], result["leak_pct"]
        print(f"=== {role}: {result['examples']} examples, "
              f"{result['train']} train / {result['val']} val ===")
        print(f"  corpus: {result['corpus']['source_files']} source files, "
              f"{result['corpus']['phrase_families']} phrase families, "
              f"{result['corpus']['duplicate_examples']} duplicate examples")
        for key in ("byte_identical_twin_in_train", "same_source_file_in_train",
                    "same_phrase_family_in_train", "clean"):
            print(f"  val examples with {key:<32} {leak[key]:>5}  ({pct[key]:>5.1f}%)")
        held = result["families_held_out_entirely"]
        print(f"  phrase families held out entirely: {len(held)} {held}")

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
