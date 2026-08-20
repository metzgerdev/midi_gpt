"""Build the training corpus: a folder of MIDI in, .npz training examples out.

ARCHIVAL. Needs a multitrack MIDI collection this repo does not ship, since the corpus is
a licensed sample-pack derivative. To adapt the models, fine-tune the checkpoints with
`python -m train.finetune_dpo` instead.

Each example is one 4-bar chunk of a single melodic line, reduced to the representation
the model is trained on:

    tokens (64,)      one per sixteenth: a pitch, a sustain, or a rest
    grid   (64,)      onset positions — the rhythm skeleton
    chord  (64, 12)   pitch classes of the chord — the harmony skeleton

Both conditioning tracks are self-supervised. The grid is every one of the stem's note-ons
— including a note re-struck at the same pitch, which `notes_to_tokens` can only spell as
a SUSTAIN, so the grid is the one place that attack survives. The chroma is inferred per
bar by matching a pitch-class histogram against chord templates. Nothing is annotated by
hand.

Every chunk is written twelve times, transposed through all keys, with the notes and the
chroma rotated together. That is the point of the augmentation: across the twelve copies
the rhythm is identical and only the chroma predicts which pitches appear, so the model
cannot use absolute pitch and must read the chord track instead.

    python -m train.mine_corpus --corpus ~/path/to/midi                 # bass
    python -m train.mine_corpus --corpus ~/path/to/midi --role arp
    python -m train.mine_corpus --corpus ~/path/to/midi --filter all    # skip the genre filter
"""

from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import mido
import numpy as np

from utils.chords import infer_chord_track
from utils.config import (
    GRID_STEPS_PER_BAR, NOTE_BARS, NOTE_PITCH0, NOTE_STEPS, ROLE_CENTER, ROLE_MONO,
    TRAINING_DATA,
)
from utils.midi_utils import midi_to_step_grid, note_onsets, notes_to_tokens, octave_fit


# Which filenames belong to which role. Bass is one pattern; the melodic role covers the
# several names a multitrack corpus tends to use for the same thing.
ROLE_PATTERNS = {
    "bass": ("*bass*",),
    "arp": ("*arp*", "*pluck*", "*lead*", "*melody*", "*keys*"),
}
GENRE_KEYWORDS = ("ukg", "garage", "2step", "2-step")


def select_files(corpus: Path, role: str, genre_filter: str, limit: int | None):
    """Filenames matching the role, optionally narrowed to a genre by path."""
    files = []
    for pattern in ROLE_PATTERNS[role]:
        files += glob.glob(str(corpus / "**" / f"{pattern}.mid*"), recursive=True)
    if genre_filter != "all":
        files = [f for f in files if any(k in f.lower() for k in GENRE_KEYWORDS)]
    return sorted(set(files))[:limit]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="folder of MIDI to mine, searched recursively")
    ap.add_argument("--role", default="bass", choices=sorted(ROLE_PATTERNS))
    ap.add_argument("--filter", default="ukg", choices=("ukg", "all"),
                    help="'ukg' keeps paths mentioning ukg/garage/2step (default)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: training_data/<role>_notes_midi)")
    ap.add_argument("--min-notes", type=int, default=3,
                    help="drop chunks with fewer onsets than this")
    ap.add_argument("--key-aug", type=int, default=12,
                    help="transposition copies per chunk; 12 covers every key")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if not args.corpus.is_dir():
        raise SystemExit(f"no such corpus directory: {args.corpus}")
    out = args.out or TRAINING_DATA / f"{args.role}_notes_midi"
    out.mkdir(parents=True, exist_ok=True)
    mono, center = ROLE_MONO[args.role], ROLE_CENTER[args.role]

    files = select_files(args.corpus, args.role, args.filter, args.limit)
    print(f"{len(files)} {args.role} files matched in {args.corpus}"
          f"{'' if args.filter == 'all' else ' (genre-filtered)'}", flush=True)

    seen, written, duplicates, unreadable = set(), 0, 0, 0
    for path in files:
        # dedup on content: the same stem is often copied across many project folders
        digest = hashlib.md5(Path(path).read_bytes()).hexdigest()[:10]
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        try:
            midi = mido.MidiFile(path)
        except Exception:
            unreadable += 1
            continue

        pitch = midi_to_step_grid(midi, mono=mono)
        if pitch is None:
            continue
        onsets = note_onsets(midi, len(pitch))

        stem = Path(path).stem[:28]
        for chunk in range(len(pitch) // NOTE_STEPS):
            lo, hi = chunk * NOTE_STEPS, (chunk + 1) * NOTE_STEPS
            segment, grid = pitch[lo:hi], onsets[lo:hi]
            centred = octave_fit(segment, 0, center)
            if int((notes_to_tokens(centred) >= NOTE_PITCH0).sum()) < args.min_notes:
                continue
            # chroma is inferred once, from the un-transposed chunk, then rotated with it
            chroma = infer_chord_track(centred, NOTE_BARS, GRID_STEPS_PER_BAR)
            for shift in range(args.key_aug):
                tokens = notes_to_tokens(octave_fit(segment, shift, center))
                np.savez(out / f"{stem}_{digest}__{chunk}_k{shift}.npz",
                         tokens=tokens, grid=grid,
                         chord=np.roll(chroma, shift, axis=1).astype(np.float32))
                written += 1

    print(f"{written} examples from {len(seen)} unique files "
          f"({duplicates} duplicates, {unreadable} unreadable) -> {out}")
    if written:
        print(f"that is {written // args.key_aug} distinct chunks x {args.key_aug} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
