# training_data

Everything the note models learned from. Inference does not read any of it — this
exists so the models can be retrained or the preference flywheel continued.

```
training_data/
├── bass_notes_midi/   4248 .npz   mined bass segments
├── arp_notes_midi/    3456 .npz   mined arp/lead segments
├── dpo/               5 tracks    hand-edited preference pairs
└── drum_loops/        1 .wav      the loop those pairs were conditioned on
```

## Mined segments

Each `.npz` holds one 4-bar example: `tokens (64,)`, `grid (64,)`, `chord (64, 12)`.
Filenames end `_k0` … `_k11` — twelve transpositions of the same source segment, which
is what forces the model to read the chord track instead of memorising absolute pitch.
So the real counts are **354 unique bass segments** and **288 unique arp segments**.

Both conditioning signals are self-supervised: the onset grid comes from the stem's own
note onsets at training time (swapped for a real kick grid at inference), and the chord
chroma is inferred per bar by template matching rather than labelled.

Mined from a multitrack MIDI corpus of 19,356 files
after content-hash
deduplication. **Provenance caveat: about 90% of the sources are that project's own
generated output** — 317 of 354 bass sources are generically named `bass_<hash>`, and
only 37 carry descriptive names from a UKG sample pack. The models are therefore
largely trained on another model's output.

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

Everything needed is in the repository:

```bash
uv run --frozen python -m train.mine_corpus --corpus <folder of MIDI> --role bass
uv run --frozen python -m train.mine_corpus --corpus <folder of MIDI> --role arp
uv run --frozen python -m train.train_notes --role bass
uv run --frozen python -m train.train_notes --role arp
```

`train_notes` refuses to run if its target checkpoint already exists — retraining would
replace the weights every other script loads by default. Pass `--out <path>` to write
elsewhere, or `--force` to overwrite.


The corpus these examples came from is a multitrack MIDI collection outside this
project; `mine_corpus.py` takes its location as `--corpus` rather than assuming it.
