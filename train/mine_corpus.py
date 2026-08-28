"""Build the training corpus: a folder of MIDI in, .npz training examples out.

Reads `training_data/corpus/` by default. That tree is vendored but gitignored — it is
purchased pack MIDI verbatim, and the licence covers using the packs, not redistributing
them. What is tracked is `training_data/manifests/<role>.json`, which records the funnel
and maps every example's content digest back to the named source file it came from.

Provenance is the reason the manifest exists. A multitrack corpus names the folder after
the arrangement and the file after the role, so the source of an example is `bass.mid` —
89% of the first corpus mined here were named exactly that, and the .npz filename keeps
only the first 28 characters of it. Nothing downstream could say what any example was.

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

Mine an *un-augmented* corpus. If the source is already transposition-augmented, content
hashing will not catch it — transposing changes every byte — and the twelve keys of one
phrase enter as twelve phrases, then get twelve keys each. The `family` field in the
manifest is what makes that visible, and what a validation split must group on.

    python -m train.mine_corpus                                # bass, vendored corpus
    python -m train.mine_corpus --role arp
    python -m train.mine_corpus --filter all                   # skip the genre filter
    python -m train.mine_corpus --corpus ~/path/to/other/midi
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
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

# A multitrack corpus names the *folder* after the arrangement and the *file* after the
# role, so `bass.mid` alone identifies nothing — 317 of the first corpus mined here were
# named exactly that. The folder is the phrase identity, and a transposed copy of one
# phrase is the same phrase: an augmented corpus tags those `_t+03`, `_t-05`. Content
# hashing cannot see through that, because transposing changes every byte. Stripping the
# tag can, which is what keeps twelve keys of one bassline from being counted as twelve
# basslines — and what a split has to group on so they cannot straddle it.
TRANSPOSE_TAG = re.compile(r"_t[+-]?\d+$")


ROLE_STEMS = ("bass", "melody", "arp", "lead", "pluck", "keys", "harmony")


def phrase_family(path: Path) -> str:
    """The identity a source shares with its own transpositions.

    Role-named file in an arrangement folder -> the folder, minus any transposition tag.
    Descriptively-named pack file -> itself; nothing else in the corpus is that phrase.
    """
    if path.stem.lower() in ROLE_STEMS:
        return TRANSPOSE_TAG.sub("", path.parent.name)
    return TRANSPOSE_TAG.sub("", path.stem)


def resolve_families(paths_by_digest: dict[str, list[Path]]) -> dict[str, str]:
    """Group digests into phrases, merging names that share a byte-identical file.

    Two things hide the same phrase behind different names, and neither survives a rule
    that looks at one path. Transposition changes every byte, so the twelve keys of a
    bassline get twelve digests; and one loop is reused across arrangements, so each of
    those twelve may be found under a different folder. Naming a digest after the single
    path that won deduplication therefore splits one phrase into several families —
    65 instead of 45 on this corpus, which would understate a validation leak by a third.

    So: union a digest with every name any copy of it carries, and union those names with
    each other. Whatever ends up connected is one phrase, labelled by its lowest name.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for digest, paths in paths_by_digest.items():
        node = f"digest:{digest}"
        find(node)
        for path in paths:
            union(node, f"name:{phrase_family(path)}")

    labels: dict[str, set[str]] = {}
    for digest, paths in paths_by_digest.items():
        root = find(f"digest:{digest}")
        labels.setdefault(root, set()).update(phrase_family(p) for p in paths)
    return {digest: min(labels[find(f"digest:{digest}")]) for digest in paths_by_digest}


def select_files(corpus: Path, role: str, genre_filter: str, limit: int | None):
    """Filenames matching the role, optionally narrowed to a genre by path.

    Also returns the funnel counts, because "4,248 examples" says nothing about the
    19,356 files that were not chosen, and the drop at each stage is the part of the
    dataset story a reader cannot reconstruct from the output directory.
    """
    scanned = sum(1 for _ in corpus.rglob("*.mid*"))
    files = []
    for pattern in ROLE_PATTERNS[role]:
        files += glob.glob(str(corpus / "**" / f"{pattern}.mid*"), recursive=True)
    matched = sorted(set(files))
    kept = matched
    if genre_filter != "all":
        kept = [f for f in matched if any(k in f.lower() for k in GENRE_KEYWORDS)]
    limited = kept[:limit]
    return limited, {
        "scanned": scanned,
        "role_matched": len(matched),
        "genre_kept": len(kept),
        "after_limit": len(limited),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=TRAINING_DATA / "corpus",
                    help="folder of MIDI to mine, searched recursively "
                         "(default: training_data/corpus)")
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
    ap.add_argument("--manifest", type=Path, default=None,
                    help="where to write the provenance manifest "
                         "(default: training_data/manifests/<role>.json)")
    args = ap.parse_args(argv)

    if not args.corpus.is_dir():
        raise SystemExit(
            f"no such corpus directory: {args.corpus}\n"
            "training_data/corpus/ is gitignored pack MIDI and is absent from a fresh "
            "clone. training_data/corpus/README.md records what belongs there; "
            "training_data/manifests/ lists every file the shipped corpus was mined from."
        )
    out = args.out or TRAINING_DATA / f"{args.role}_notes_midi"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or TRAINING_DATA / "manifests" / f"{args.role}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mono, center = ROLE_MONO[args.role], ROLE_CENTER[args.role]

    files, funnel = select_files(args.corpus, args.role, args.filter, args.limit)
    print(f"{len(files)} {args.role} files matched in {args.corpus}"
          f"{'' if args.filter == 'all' else ' (genre-filtered)'}", flush=True)

    # Hash first, mine second. Every copy of a file has to be seen before a phrase can be
    # named, because the copy that wins deduplication is not always the one whose folder
    # names the phrase — see resolve_families.
    paths_by_digest: dict[str, list[Path]] = {}
    for path in files:
        # dedup on content: the same stem is often copied across many project folders
        digest = hashlib.md5(Path(path).read_bytes()).hexdigest()[:10]
        paths_by_digest.setdefault(digest, []).append(Path(path))
    duplicates = len(files) - len(paths_by_digest)
    family_of = resolve_families(paths_by_digest)

    written, unreadable = 0, 0
    no_notes, too_short, sparse_chunks, sources = 0, 0, 0, []
    for digest, copies in paths_by_digest.items():
        path = copies[0]
        try:
            midi = mido.MidiFile(path)
        except Exception:
            unreadable += 1
            continue

        pitch = midi_to_step_grid(midi, mono=mono)
        if pitch is None:
            no_notes += 1
            continue
        if len(pitch) < NOTE_STEPS:
            too_short += 1
            continue
        onsets = note_onsets(midi, len(pitch))

        source = path
        stem = source.stem[:28]
        chunks_kept = 0
        for chunk in range(len(pitch) // NOTE_STEPS):
            lo, hi = chunk * NOTE_STEPS, (chunk + 1) * NOTE_STEPS
            segment, grid = pitch[lo:hi], onsets[lo:hi]
            centred = octave_fit(segment, 0, center)
            if int((notes_to_tokens(centred) >= NOTE_PITCH0).sum()) < args.min_notes:
                sparse_chunks += 1
                continue
            # chroma is inferred once, from the un-transposed chunk, then rotated with it
            chroma = infer_chord_track(centred, NOTE_BARS, GRID_STEPS_PER_BAR)
            for shift in range(args.key_aug):
                tokens = notes_to_tokens(octave_fit(segment, shift, center))
                np.savez(out / f"{stem}_{digest}__{chunk}_k{shift}.npz",
                         tokens=tokens, grid=grid,
                         chord=np.roll(chroma, shift, axis=1).astype(np.float32))
                written += 1
            chunks_kept += 1

        # The .npz filename keeps the digest but truncates the stem to 28 chars, and for
        # a role-named file that stem is "bass" — so the manifest is the only place the
        # example can be traced back to a named source. Written even when the file
        # yielded nothing, so the funnel adds up.
        sources.append({
            "digest": digest,
            "path": str(source.relative_to(args.corpus)) if args.corpus in source.parents
                    else str(source),
            "family": family_of[digest],
            "copies": len(copies),
            "chunks": chunks_kept,
            "examples": chunks_kept * args.key_aug,
        })

    kept_sources = [s for s in sources if s["examples"]]
    families = {s["family"] for s in kept_sources}
    funnel.update({
        "content_duplicates": duplicates,
        "unreadable": unreadable,
        "no_note_data": no_notes,
        "shorter_than_4_bars": too_short,
        "chunks_below_min_notes": sparse_chunks,
        "unique_sources": len(kept_sources),
        "distinct_phrase_families": len(families),
        "examples_written": written,
    })
    manifest_path.write_text(json.dumps({
        "role": args.role,
        "corpus": str(args.corpus),
        "mined_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "options": {"filter": args.filter, "min_notes": args.min_notes,
                    "key_aug": args.key_aug, "limit": args.limit},
        "funnel": funnel,
        "sources": sorted(kept_sources, key=lambda s: (s["family"], s["digest"])),
    }, indent=2) + "\n")

    for stage, count in funnel.items():
        print(f"  {stage:<24} {count}")
    print(f"{written} examples from {len(kept_sources)} unique files "
          f"({duplicates} duplicates, {unreadable} unreadable) -> {out}")
    print(f"{len(families)} distinct phrase families -> {manifest_path}")
    if len(kept_sources) != len(families):
        crowded = Counter(s["family"] for s in kept_sources)
        worst = ", ".join(f"{f} x{n}" for f, n in crowded.most_common(3) if n > 1)
        print(f"  note: {len(kept_sources) - len(families)} sources are transposed or "
              f"duplicate views of a family already present ({worst})")
    if written:
        print(f"that is {written // args.key_aug} distinct chunks x {args.key_aug} keys, "
              f"drawn from {len(families)} phrases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
