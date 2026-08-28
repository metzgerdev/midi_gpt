# training_data

Everything the note models learned from. Inference does not read any of it — this
exists so the models can be retrained or the preference flywheel continued.

```
training_data/
├── corpus/            19065 .mid  the source MIDI, gitignored (see its README)
├── manifests/         2 .json     digest -> source file, tracked
├── bass_notes_midi/   4248 .npz   mined bass segments
├── arp_notes_midi/    3456 .npz   mined arp/lead segments
├── dpo/               5 tracks    hand-edited preference pairs
└── drum_loops/        1 .wav      the loop those pairs were conditioned on
```

## Mined segments

Each `.npz` holds one 4-bar example: `tokens (64,)`, `grid (64,)`, `chord (64, 12)`.
Filenames end `_k0` … `_k11` — twelve transpositions of the same source segment, which
is what forces the model to read the chord track instead of memorising absolute pitch.

Both conditioning signals are self-supervised: the onset grid comes from the stem's own
note onsets at training time (swapped for a real kick grid at inference), and the chord
chroma is inferred per bar by template matching rather than labelled.

## The funnel

`mine_corpus.py` prints this and records it under `funnel` in each manifest.

| stage                                   | bass  | arp   |
| --------------------------------------- | ----- | ----- |
| `.mid` files in `corpus/`               | 19,065| 19,065|
| filename matches the role pattern       | 6,561 | 6,355 |
| path mentions `ukg`/`garage`/`2step`    | 1,483 | 1,477 |
| survives content-hash dedup             | 393   | 379   |
| at least 4 bars long                    | 354   | 289   |
| chunk has ≥ 3 sounding notes            | 354   | 288   |
| **examples written** (× 12 keys)        | 4,248 | 3,456 |
| **distinct phrase families**            | 45    | 45    |

## Provenance

Every example traces to a named source file. `manifests/<role>.json` maps each content
digest to the path it was mined from, because the `.npz` filename cannot: it keeps the
first 28 characters of the source stem, and for a multitrack corpus that stem is the
role. 317 of the 354 bass sources are named exactly `bass.mid`, 251 of 288 arp sources
`melody.mid`. Resolved through the manifest, 37 of the 45 bass families reach a
descriptively-named pack loop (`SO_UKG_140_bass_reesy_Cmin.mid`) and the remaining 8
reach a `SOUKG_<key>_<index>` arrangement folder from the same pack. Nothing is unknown.

**Correcting an earlier claim in this file.** It used to say about 90% of the sources
were "that project's own generated output" and that the models were "largely trained on
another model's output." That was inferred from the generic filenames and it is wrong.
Hashing all 19,356 files of the source project against the 642 corpus digests matches
every one, and **none** of them resolve to that project's `output/` directory — they
come from `data/genre_corpus_aug/`, which is its *input* corpus, assembled from
purchased packs. The models are trained on sample-pack MIDI, not on model output.

**A real caveat, in its place.** The corpus was mined from an already
transposition-augmented tree, and content hashing cannot see through a transposition —
it changes every byte. So the 354 and 288 "unique" sources are **45 phrase families
each**, present in twelve keys, which `mine_corpus.py` then transposes twelve more
times. A third of the bass corpus (1,420 of 4,248) and 42% of the arp corpus are
bit-identical duplicates. The `family` field in each manifest is what makes this
visible, and a validation split has to group on it: a split that does not put every key
of a phrase on the same side is measuring memorisation, not generalisation.

Mining `corpus/genre_corpus/` instead of `corpus/genre_corpus_aug/` yields the same 45
families as 45 sources and 540 examples with no duplicates. The shipped checkpoints were
trained on the 4,248, so the larger corpus is kept to make them reproducible.

## DPO pairs

Five tracks, each holding the conditioning it was generated under plus both sides of
the preference pair:

```
<track>/metadata.json          progression, bpm, drum loop, seed, weights
<track>/stems/<role>.mid       REJECTED — what the model produced
<track>/stems/<role>_edited.mid  CHOSEN — after hand-editing in Ableton
```

`metadata.json` is not optional. The trainers rebuild the exact conditioning from it:
the progression becomes the chord chroma, and `drum_loop` names the wav whose kick
pocket becomes the rhythm grid. All five reference `drumgen_1_temp0.7.wav`, kept in
`drum_loops/` for that reason — the copy under `stem_renders/` is not safe to rely on.

These 20 files are the only human-authored signal anywhere in the project.

Two of the ten role-pairs are identical in note space (`a_minor_dark_8bar_ft1` arp and
`c_minor_dark_8bar` arp): those edits were velocity or sub-grid timing, which the
16th-note token representation cannot express. They contribute nothing to a preference
objective and get skipped. Eight pairs carry real signal.

## Rebuilding

`mine_corpus.py` defaults to `corpus/`, so on a machine that has it:

```bash
uv run --frozen python -m train.mine_corpus --role bass
uv run --frozen python -m train.mine_corpus --role arp
uv run --frozen python -m train.train_notes --role bass
uv run --frozen python -m train.train_notes --role arp
```

Mining is pinned: rerunning the first two commands rewrites all 7,704 `.npz` with
byte-identical contents, and `test_manifest_accounts_for_every_shipped_example` fails if
the manifests and the mined files disagree. Point `--corpus` elsewhere to mine a
different collection.

`train_notes` refuses to run if its target checkpoint already exists — retraining would
replace the weights every other script loads by default. Pass `--out <path>` to write
elsewhere, or `--force` to overwrite.

`corpus/` is gitignored, so a fresh clone has the manifests but not the MIDI. The
manifests are enough to say what every example is; they are not enough to rebuild it.
