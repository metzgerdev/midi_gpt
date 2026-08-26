# MIDI GPT
Small language model trained on house and uk garage patterns.  Provide a drum loop, bpm and optionally a chord progression, the model will generate 8 bar bass and melody pattern. The model is currently tuned to my taste, however a DPO (Direct Preference Optimization) script allows finetuning so that it reflects your taste.  

# Why I built this model
As a music producer, sitting down and starting a blank DAW session can feel insurmountable.  Music is built from small beginnings, and a short catchy loop can be just enough to get creativity flowing.  Even better if the midi generated has been tuned to your taste.  This is a small parameter model, runs locally, and generates midi in a couple of seconds. I chose midi generation to preserve a human in the loop control, and also for maximum sound fidelity since audio is created by the DAW.  

# What the models were trained on

Both note models learned from MIDI mined out of a multitrack house and UK garage
collection: bass stems for the bass model, and arp/pluck/lead/melody/keys stems for the
melodic one.

Each training example is one 4-bar phrase, reduced to a single voice and written as one
token per sixteenth note — strike a pitch, hold the last one, or rest. There is no
velocity, no polyphony, and no duration field: a note's length is just how many holds
follow it.

Alongside every phrase sit the two signals the model is conditioned on, both derived from
the phrase itself rather than labelled by hand:

- **grid** — where the notes land, a rhythm skeleton
- **chord** — which pitch classes are sounding, one chord per bar, a harmony skeleton

Every phrase is stored in all twelve keys, with the notes and the chord rotated together.
That is what forces the model to read the chord track: across the twelve copies the rhythm
is identical, so only the chord predicts which pitches appear.

|                        | bass | arp |
| ---------------------- | ---- | --- |
| distinct 4-bar phrases | 72   | 66  |
| distinct rhythms       | 40   | 39  |
| keys per phrase        | 12   | 12  |
| steps per phrase       | 64   | 64  |

This is a small corpus, and a narrow one by choice — one genre, one producer's selection.
The model plays UK garage because that is what it has heard. If its taste is not yours,
that is what the DPO fine-tuning is for.

# How I trained the model

`train_notes.py`, from random initialization. There is no pretraining stage; the model is
trained directly on the target distribution.

The objective is next-token cross-entropy. Each example becomes `[BOS] + 64 tokens +
[EOS]`; the input is that sequence minus its last token and the target is it minus its
first, so every position predicts the next. Training is teacher-forced: one forward pass
covers all 65 positions at once, with a causal mask preventing lookahead. The grid and
chroma are supplied at every position, aligned to the step being predicted.

The loss is class-weighted:

```python
w[NOTE_SUSTAIN]  = 0.3
w[pitch tokens]  = 3.0
```

A bassline is mostly sustain and rest, and unweighted cross-entropy learns to predict
those and emit almost no notes. Upweighting onsets is what makes the model play.

AdamW, learning rate 5e-4, weight decay 0.05, checkpointing on best validation loss.
Final: 0.132 for bass, 0.097 for arp.

Two fine-tuning rounds follow, both starting from the base checkpoint and mixing the edits
with a replayed sample of the base corpus at a tenth the learning rate so the base
distribution survives: `_ft1` on 4 hand-edited tracks, `_ft2` on 10. `_ft3` is the DPO
stage below.

# What the model predicts

At each of the 64 steps the model's input is the sum of four vectors, all 128-dimensional:

```
x = token embedding      what was played at this step
  + position embedding   where this step falls in the four bars
  + grid projection      whether a kick lands here          Linear(1, 128)
  + chroma projection    which chord is sounding            Linear(12, 128)
```

They are summed rather than concatenated, which keeps the model 128 dimensions wide.
Training pushes each signal into its own directions, so the sum stays separable — any two
of the four are as unrelated as randomly chosen vectors.

The chroma is a 12-dimensional multi-hot mask, so its projection is the sum of one learned
column per pitch class in the chord. Chords therefore compose additively: a seventh is the
triad's vector plus one more column, which is why progressions the model never saw during
training still work.

The output is 65 logits, and the sampled token is exactly one of:

```
0        REST      end whatever was sounding
1        SUSTAIN   hold the previous note one more step
2..62    pitch     strike a new note, MIDI 24..84
63, 64   BOS, EOS
```

Sampling uses temperature and nucleus (top-p 0.98). A note's length is emergent — one step
plus however many SUSTAIN tokens follow it — so the representation has no duration field,
no velocity and no polyphony. `midi_utils.py` converts the finished token sequence to
`(pitch, start_step, duration)` triples and writes the MIDI at 480 ticks per beat, 120 per
sixteenth, with a fixed velocity of 90.

# How to fine tune using DPO

The loop is: generate, edit the MIDI in a DAW, feed both versions back.

Export the edited clip into the run's own folder as `stems/<role>_edited.mid`, leaving the
original `stems/<role>.mid` in place. The pair is the training signal — the original is
the rejected sample, your edit is the chosen one — and because both sit in the run folder,
`metadata.json` supplies the exact conditioning they were generated under.

`finetune_dpo.py` reconstructs that conditioning, tokenizes both sides, augments through
twelve keys, and optimizes:

```
r(seq)  = beta * ( logP_policy(seq) - logP_anchor(seq) )
margin  = r(chosen) - r(rejected)
loss    = -log sigmoid(margin) + lambda * KL(policy || anchor)
```

The policy starts at the highest `ftN`; the anchor is a *different*, fixed checkpoint —
by default the untuned base, so drift is measured from one place across rounds rather
than compounding. `beta` only scales the reward and cannot bound drift on its own. The
KL term does, with `lambda` steered every step to hold `--kl-target`. Defaults are beta
0.1, learning rate 1e-5, 15 epochs, KL budget 0.05 per run. Because chosen and rejected
are both exactly 64 steps, the length bias that affects DPO on variable-length text
cannot occur here.

Pairs that tokenize identically are skipped. Velocity and sub-grid timing edits fall into
this category — the representation cannot express them — so edits that change *which
notes are played where* are the ones that count.

The output is a new `<role>_notes_gpt_ftN.pt`, and `make_track.py` selects the highest
`ftN` automatically, so the next batch you edit is drawn from the model that already
absorbed the previous round. `--base` runs the untuned models as a control.

The current checkpoints were built this way: `_ft3` is DPO from `_ft2` with 10 bass pairs
and 6 arp pairs at beta 0.1.

In practice:

```bash
# 1. generate
uv run --frozen python -m inference.make_track --request "8 bar UKG track in A minor, dark"

# 2. open output/<run>/stems/bass.mid in a DAW, change it, and export it back
#    into the same folder as bass_edited.mid, leaving bass.mid in place

# 3. see what would be learned, without training
uv run --frozen python -m train.finetune_dpo --dry-run

# 4. train
uv run --frozen python -m train.finetune_dpo
uv run --frozen python -m train.finetune_dpo --role bass --beta 0.2 --epochs 20
```

The run reports a reward margin and preference accuracy, then the two halves the margin
is built from — the chosen and rejected log-ratios against the anchor. Watch the halves.
The margin is the quantity being optimized, measured on the very pairs it trained on, so
it rises close to by construction; what it cannot show is *how*. DPO can win it by making
your edit more likely or by making the model's own output less likely, and since the
rejected sample was drawn from the checkpoint the policy starts at, that output is a
bassline the model would ordinarily play. The run warns when the chosen side does not
rise at all, or when more than 70% of the gain came from pushing the rejected side down.

If the mined corpus is present it also reports the loss on it before and after, which
should barely move. That one is genuinely held out from the pairs, so it is the real
regression guard. A large jump means the model drifted off what it already knew: lower
`--kl-target` or cut `--epochs`. Raising `--beta` will not help.

`finetune_sft.py` is the lighter alternative: it trains on the edit alone rather than
the pair, so it needs no original and pulls toward what you kept without pushing away
from what you replaced.

# Retraining from scratch

The whole path is in the repository. Nothing here is needed to generate — only to
rebuild the models.

```bash
# 1. corpus of MIDI -> training examples
uv run --frozen python -m train.mine_corpus --corpus ~/path/to/midi --role bass
uv run --frozen python -m train.mine_corpus --corpus ~/path/to/midi --role arp

# 2. examples -> base checkpoints
uv run --frozen python -m train.train_notes --role bass
uv run --frozen python -m train.train_notes --role arp

# 3. optional: fine-tune on your edits
uv run --frozen python -m train.finetune_sft
uv run --frozen python -m train.finetune_dpo
```

`train_notes` refuses to run if its target checkpoint already exists — retraining would
replace the weights every other script loads by default. Pass `--out <path>` to write
elsewhere, or `--force` to overwrite.


`mine_corpus.py` searches recursively for filenames matching the role — `*bass*`, or
`*arp* *pluck* *lead* *melody* *keys*` — and by default keeps only paths mentioning
`ukg`, `garage` or `2step`. Pass `--filter all` to mine everything it finds. Files are
deduplicated on content, since the same stem tends to appear in many project folders.

Training runs on the CPU in minutes at this scale. The models are small enough that the
binding constraint is corpus size, not compute.


`make_track.py` parses a text request into a key, mood and length, derives a chord
progression from them, and conditions two small GPT note models on that progression
and on the kick pattern of a drum loop. Each model emits one token per sixteenth
note — a pitch, a sustain, or a rest — which becomes a MIDI file.

Output is symbolic. The drum loop is read only for its kick-pocket onset grid; its
audio conditions generation and is never mixed into the output. Load the generated
MIDI into a DAW to voice it.

## How it works

Two skeletons condition every step of generation:

- **Rhythm** — the drum loop is low-passed at 160 Hz, reduced to an onset-strength
  envelope, and binned into a binary kick-per-sixteenth grid.
- **Harmony** — the progression becomes a 12-dimensional pitch-class mask per bar.

Both are projected to the model's width and added into its embeddings at each step,
so the model chooses notes inside a rhythm and a harmony it is given rather than
inventing them. A run generates independent 4-bar sections and concatenates them,
scoring several candidates per section and keeping the best by chord-tone fit,
kick-lock and note density.

The note models are 0.687M parameters each: 65-token vocabulary, 3 layers, 128
dimensions, 4 heads.

`pipeline.html` is a standalone diagram of the whole path.

## Requirements

- macOS with Python 3.12
- [uv](https://docs.astral.sh/uv/)
- `checkpoints/bass_notes_gpt_ft3.pt` and `checkpoints/arp_notes_gpt_ft3.pt`
- at least one 2-step drum loop in `drum_samples/`

## Setup

```bash
uv sync --frozen
```

## Generate

Interactive — prompts for genre, key, tempo, bars, and whether to use the fine-tuned
weights. A bare invocation in a terminal does this; `-i` / `--interactive` forces it
when other arguments are present.

```bash
uv run --frozen python -m inference.make_track
```

From a request:

```bash
uv run --frozen python -m inference.make_track \
  --request "8 bar UKG track in A minor, dark" \
  --seed 7
```

Controls:

```bash
# Untuned checkpoints, as a control
uv run --frozen python -m inference.make_track --base

# Bass only
uv run --frozen python -m inference.make_track --no-arp

# Genre preset and tempo
uv run --frozen python -m inference.make_track --genre ukg-classic --bpm 138

# A different drum loop
uv run --frozen python -m inference.make_track --drum path/to/loop.wav --drum-bpm 127

# Either role's checkpoint
uv run --frozen python -m inference.make_track \
  --bass-ckpt checkpoints/bass_notes_gpt_ft3.pt \
  --arp-ckpt checkpoints/arp_notes_gpt_ft3.pt

# Somewhere other than output/
uv run --frozen python -m inference.make_track --out-root /tmp/takes

# Sample on the GPU
uv run --frozen python -m inference.make_track --device auto

# Reproduce a run: its seed, on the same device
uv run --frozen python -m inference.make_track --seed 424242

# More candidates per section
uv run --frozen python -m inference.make_track --candidates 20

# Looser sampling
uv run --frozen python -m inference.make_track --bass-temp 1.5 --arp-temp 1.6

# In-key 7ths and suspensions, drawn fresh per section
uv run --frozen python -m inference.make_track --chords color
```

`inference.ipynb` runs the same pipeline in a notebook and shows the generated notes
against the kick grid.

## Output

Every run writes its own folder:

```
output/<MMDD-HHMMSS>_<key>_<mood>_<bars>bar[_<bpm>bpm][_ftN]/
    stems/bass.mid
    stems/arp.mid
    metadata.json
```

The timestamp is to the second, so runs are uniquely named, sort chronologically, and
never overwrite each other. An `_ftN` tag records that fine-tuned weights ran; its
absence means `--base`. `metadata.json` holds the progression per section, sampling
settings, device, the drum loop and its tempo, and per-role validation scores —
chord-tone fit and kick-lock.

## Behavior worth knowing

**Seed.** Each run draws a new random seed unless `--seed` is given, and prints it.
Generation is otherwise deterministic, so a seed reproduces a run byte for byte.

Seed is the diversity control; temperature is a weak one. Across six runs of one
section, varying the seed moves 43% of steps, while raising bass temperature from 1.2
to 1.8 reaches only 52% and costs kick-lock 0.87 → 0.61. Lowering `--candidates`
widens variation as well (0.43 → 0.54 at N=1) but drops chord fit 0.74 → 0.66, since
best-of-N is what holds the quality floor.

**Device.** Sampling runs on the CPU by default. Generation is 64 sequential
single-token passes through a 0.687M-parameter model, and at that size accelerator
launch overhead costs more than the arithmetic saves: ~632 ms per candidate on MPS
against ~60 ms on CPU, so an 8-bar run takes about 25 s on the GPU and 2.4 s on the
CPU. `--device auto` selects an accelerator if one is present.

A seed reproduces on the same device only — MPS and CPU diverge within a few steps
from identical state — so `metadata.json` records which device sampled the run.

**Chord coloring.** `--chords color` keeps each chord's root and major/minor quality
and draws a 7th or suspension for it, admitting a candidate only when all of its tones
are diatonic. Staying in key is therefore a property of the key rather than a curated
table: A minor admits `Fmaj7` on the VI and `G7` on the VII while rejecting `F7` (E♭)
and `Gmaj7` (F♯). Sections are colored independently, so an 8-bar run is not the same
four bars of harmony twice.

The model uses it: with 7ths, 37% of generated notes land on tones the plain triad does
not contain, and 25% with suspensions. `chord_fit` rises under four-note chords simply
because there are more tones to hit, so compare it within a chord mode, not across.

**Drum loops.** The conditioning loop is whichever `.wav` sits in `drum_samples/`
(first by name when there are several), overridable with `--drum`. Put the loop's
tempo in its filename (`house_drums_loop_127bpm.wav`) or pass `--drum-bpm`: the
analysis window is a fixed 3.693 s — two bars at 130 — so a loop at another tempo has
its onsets land in the wrong sixteenth bins, and the error compounds across the bar.
On a 127 BPM house loop that misplaces 9 of 32 steps, including the downbeat. Loops
are time-stretched to 130 for analysis only; output tempo remains `--bpm`.

Only the first two bars reach the model — `onset_grid` truncates to `GRID_FRAMES` —
and they are tiled across each 4-bar section. Longer loops are accepted, and the run
prints how much of one was used.

## The training_data directory

What is on disk from the process described above. Inference reads none of it.

| Path | Contents |
|---|---|
| `bass_notes_midi/` | 4,248 mined bass `.npz` — 72 distinct phrases, each in 12 keys |
| `arp_notes_midi/` | 3,456 mined arp `.npz` — 66 distinct phrases, each in 12 keys |
| `dpo/` | 5 tracks of hand-edited MIDI preference pairs |
| `drum_loops/` | the loop those pairs were conditioned on |

The mined `.npz` sets are not tracked: they are lossy but musically recognizable
derivatives of licensed sample-pack MIDI, and are regenerable from the source corpus.
The DPO pairs are tracked — they are the only human-authored data in the project.
`training_data/README.md` documents how everything was produced and how to rebuild it.

## Source

| File | Responsibility |
|---|---|
| **`inference/`** | |
| `make_track.py` | CLI and end-to-end MIDI generation pipeline |
| `inference.ipynb` | notebook front end |
| **`model/`** | |
| `note_model.py` | conditioned GPT note model and sampler |
| `gpt_model.py` | GPT backbone |
| `checkpoints.py` | finding and loading trained checkpoints |
| **`utils/`** | |
| `config.py` | paths, token, rhythm, and conditioning constants |
| `chords.py` | chord parsing, per-step chroma, in-key coloring |
| `midi_utils.py` | conversion between note tokens and MIDI, both directions |
| `audio_features.py` | drum-loop kick-pocket onset grid, tempo alignment |
| `device.py` | device selection |
| `scoring.py` | chord-tone fit and kick lock for a generated section |
| **`train/`** | |
| `finetune_dpo.py` | fine-tune on preference pairs |
| `finetune_sft.py` | fine-tune on edits alone |
| `mine_corpus.py` | MIDI corpus -> training examples |
| `train_notes.py` | training examples -> base checkpoint |
| **`tests/`** | |
| `test_inference.py` | tokenizer, conditioning, chord and checkpoint tests |
| `test_finetune_dpo.py` | iterated DPO: distribution moves, KL stays bounded (slow) |
| **root** | |
| `pipeline.html` | standalone diagram |

## Test

```bash
uv run --frozen pytest              # fast suite, ~1.5s
uv run --frozen pytest -m slow      # iterated DPO, runs real training rounds
```

Modules are namespace packages under the repo root, so entry points run with `-m`
from the project directory:

```bash
uv run --frozen python -m inference.make_track --request "8 bar UKG track in A minor"
uv run --frozen python -m train.finetune_dpo --role bass
```

## UI

A local Streamlit front end for generating and for the fine-tuning loop.

```bash
uv run --frozen streamlit run ui/app.py
```

**Generate** maps the CLI flags to controls and shows the run's progression, seed and
validation scores, with the stems as downloads.

**Fine-tune** is the taste loop: download a stem, edit it in a DAW, upload it back, and
run DPO. It reports the reward margin and how much KL the run added against its anchor,
and lists the checkpoint chain each role has accumulated.

There is no audio playback and no piano roll — the stems go into a DAW, which does both
better.

`.streamlit/config.toml` turns off Streamlit's usage telemetry, which otherwise posts to
`webhooks.fivetran.com` and `data.streamlit.io` on every session. With it in place the UI
makes no external requests.
