# MIDI GPT

Small language model trained on house and uk garage patterns.  Provide a drum loop, bpm and optionally a chord progression, the model will generate 8 bar bass and melody pattern. The model is currently tuned to my taste, however a DPO (Direct Preference Optimization) script allows finetuning so that it reflects your taste.

<!-- DEMO: 45-second video goes here — request -> generate -> drag into a DAW -> play.
     GitHub renders .mp4 inline. This is the first thing a reader should meet. -->

Two 621k-parameter GPTs trained from scratch, no pretraining. 2.1 seconds from a typed
request to MIDI on disk, on the CPU, locally.

```bash
uv run --frozen python -m inference.make_track \
  --request "make a 2 step ukg beat in A minor, dark, 8 bars long"
```

## Results at a glance

Both metrics score a generated clip against the conditioning it was given: `chord_fit`
is the share of notes landing on chord tones, `kick_lock` the correlation between note
onsets and the kick pattern. Neither means anything without reference points, so here
are four — floor, a corpus-fitted bigram, the ablation, and human-authored MIDI.

| System (bass) | `chord_fit` | `kick_lock` | note density |
|---|---|---|---|
| Random legal tokens | 0.354 ±0.007 | 0.083 ±0.007 | 61.8 |
| Bigram over the corpus | 0.463 ±0.013 | 0.124 ±0.013 | 10.9 |
| **Model, unconditioned** (zeroed grid + chroma) | 0.472 ±0.019 | 0.186 ±0.027 | 6.2 |
| **Model (ft3)** | **0.937 ±0.010** | **0.886 ±0.015** | 14.2 |
| **Human corpus** (the training MIDI itself) | 0.904 ±0.022 | 0.780 ±0.035 | 12.3 |

n=240, best-of-10, ±1 SE clustered by phrase. Arp is in [Evaluation](#evaluation).

Read the ablation row first: it is the same model given a zeroed grid and zeroed chroma,
then scored against the real conditioning it was never shown. Conditioning takes
`chord_fit` from 0.472 to 0.937 and `kick_lock` from 0.186 to 0.886. That gap is the
whole architecture claim, and it is the reason this table exists.

Then read the last row, which is the uncomfortable one: **the model beats human-authored
MIDI on both metrics.** That is not a claim that it writes better basslines than a
producer. It means these metrics are gameable, and best-of-10 games them by construction
— the selection rule is literally `2·chord_fit + kick_lock + density`. The metrics measure
constraint satisfaction, which is what they were built for and all they are evidence of.

| | CPU | MPS |
|---|---|---|
| one 4-bar candidate | **51.8 ms** | 431.4 ms |
| end to end, 8 bars, best-of-10 | **2.11 s** | 18.1 s |

CPU beats the GPU by 8.3× here: generation is 64 sequential single-token passes through a
621k-parameter model, so accelerator launch overhead costs more than the arithmetic saves.

**Two things this project gets wrong, stated up front**, because finding them is most of
the work and burying them would be the only dishonest thing in the repo:

- The corpus is **45 distinct phrases per role, not 354** — the source was already
  transposition-augmented and content hashing cannot see through a transposition. A third
  of the training examples are byte-identical duplicates. See
  [What the models were trained on](#what-the-models-were-trained-on).
- The reported **+0.022 generalisation gap is a train-train gap**. 100% of validation
  examples have another key of their own source file in train. See
  [Evaluation](#evaluation).

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

|                        | bass  | arp   |
| ---------------------- | ----- | ----- |
| distinct 4-bar phrases | 45    | 45    |
| distinct rhythms       | 38    | 38    |
| keys per phrase        | 12    | 12    |
| steps per phrase       | 64    | 64    |
| training examples      | 4,248 | 3,456 |

This is a small corpus — smaller than the example count suggests, and smaller than this
table said until I checked. The 4,248 bass examples come from 354 source files, but those
files are 45 basslines already transposed into twelve keys before mining, and content
hashing cannot deduplicate a transposition because it changes every byte. So the twelve
keys of one phrase entered as twelve phrases and left as 144 examples. A third of the
bass corpus is bit-identical duplicates. `training_data/README.md` has the full funnel
and what it costs; `training_data/manifests/` names the source file behind every example.

It is also narrow by choice — one genre, one producer's selection. The model plays UK
garage because that is what it has heard. If its taste is not yours, that is what the DPO
fine-tuning is for.

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

## Why this representation and not REMI or note-on/note-off

One token per sixteenth is the unusual choice here, so it is worth saying what it buys
against the two standard alternatives.

| | this grid | REMI | MIDI-like |
| --- | --- | --- | --- |
| event | one token per 16th step | bar / position / pitch / duration | note-on, note-off, time-shift |
| sequence length | **fixed, always 64** | variable | variable, ~2 tokens per note |
| expresses velocity | no | yes | yes |
| expresses polyphony | no | yes | yes |
| sub-16th timing | no | quantised, finer grid available | yes |
| step ↔ conditioning | **1:1, by construction** | needs alignment | needs alignment |

The two rows in bold are the whole argument, and they are the same argument twice.

**Conditioning alignment is free.** The model is steered by a kick grid and a chord
chroma, both of which are naturally per-step. With one token per step, the conditioning
vector for position *i* is simply the *i*-th row — `x = token + position + grid + chroma`
adds four aligned vectors and nothing has to be interpolated or looked up. Under REMI or
note-on/note-off, a token is an event at an arbitrary time, so every conditioning read
becomes a lookup against a clock the sequence itself is defining as it goes. The entire
conditioning mechanism this project is about gets harder for no gain.

**Fixed length removes length bias from DPO for free.** Every sequence is exactly 64
tokens, so `log P(chosen) - log P(rejected)` compares two sums with the same number of
terms. Preference optimisation on variable-length sequences has to correct for shorter
sequences scoring higher simply by summing fewer log-probabilities; here that failure
mode cannot arise, and it cannot arise structurally rather than by a normalisation term
someone has to remember to add. For a project whose point is the preference loop, that is
worth more than the expressiveness given up.

**What it cannot say, measured rather than caveated.** The cost is real and this repo has
a number for it: of the ten hand-edited preference pairs in `training_data/dpo/`, **two
tokenize identically to the clip they were meant to improve**, because those edits were
velocity and sub-grid timing changes. The representation cannot see them, so they
contribute nothing to the preference objective and are skipped — 20% of hand-editing
effort silently discarded. That is the price of the two bold rows, and it is the kind of
limitation better stated as a measurement than as a bullet in a caveats list.

Polyphony would be the bigger loss for most music, but not for these two roles: a bassline
and a lead are monophonic by convention, and `ROLE_MONO` reduces each stem to one voice at
mining time anyway (`lowest` for bass, `highest` for arp). The representation gives up
something the target material does not use.

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

# Evaluation

Three questions, answered separately because they fail separately: does the conditioning
work, does the validation loss mean anything, and did the preference tuning learn taste
or memorise ten clips.

## Does the conditioning work? Yes, and here is the ablation

**bass** — n=240 conditioning cases, best-of-10, ±1 SE clustered over 38 phrases.

| System | `chord_fit` | `kick_lock` | note density |
|---|---|---|---|
| Random legal tokens | 0.354 ±0.007 | 0.083 ±0.007 | 61.8 |
| Bigram over the corpus | 0.463 ±0.013 | 0.124 ±0.013 | 10.9 |
| **Model, unconditioned** (zeroed grid + chroma) | 0.472 ±0.019 | 0.186 ±0.027 | 6.2 |
| **Model (ft3)** | **0.937 ±0.010** | **0.886 ±0.015** | 14.2 |
| **Human corpus** (training MIDI) | 0.904 ±0.022 | 0.780 ±0.035 | 12.3 |

**arp** — n=240 conditioning cases, best-of-10, ±1 SE clustered over 38 phrases.

| System | `chord_fit` | `kick_lock` | note density |
|---|---|---|---|
| Random legal tokens | 0.358 ±0.007 | 0.103 ±0.009 | 61.9 |
| Bigram over the corpus | 0.412 ±0.009 | 0.114 ±0.014 | 17.0 |
| **Model, unconditioned** (zeroed grid + chroma) | 0.625 ±0.029 | 0.193 ±0.025 | 14.9 |
| **Model (ft3)** | **0.938 ±0.010** | **0.810 ±0.038** | 20.8 |
| **Human corpus** (training MIDI) | 0.947 ±0.013 | 0.800 ±0.042 | 20.1 |

Four things worth reading off these:

- **The ablation is decisive.** Zeroing the conditioning costs the bass model 0.465
  `chord_fit` and 0.700 `kick_lock`. The conditioning is not decoration.
- **The bigram is barely above random on `kick_lock`** (0.124 vs 0.083) while being
  clearly above it on `chord_fit`. That is the expected shape: a bigram can learn which
  notes tend to follow which, and cannot learn anything about a rhythm it is not shown.
- **The model beats the human corpus on bass, on both metrics.** The metrics are gameable
  and best-of-10 games them: the selection rule is `2·chord_fit + kick_lock + density`,
  so the model is scored on the objective it was selected under, and the corpus is not.
  On arp, where the model and corpus are within overlapping error bars, the same caveat
  applies in the other direction — that is not evidence of parity either.
- **Random has a note density of ~62 out of 64 steps.** Uniform sampling over a
  61-pitch vocabulary almost never draws REST or SUSTAIN, so the floor row is dense
  nonsense. It is the correct floor for `chord_fit` and `kick_lock` and a reminder that
  neither metric penalises playing constantly.

Reproduce with `PYTHONPATH=. python experiments/baselines/run.py`; raw output in
`experiments/baselines/baselines.json`. Every system is scored on the same conditioning
cases with the same best-of-10 rule `make_track` uses, and intervals are clustered by
phrase — the twelve keys of one phrase are one sample, not twelve.

The rows are chosen to bracket the metric rather than flatter it:

- **random** is the floor. Its `chord_fit` sits near the share of pitch classes a chord
  occupies, which is what a metric with no model behind it should give you.
- **markov** is a bigram fitted on the corpus. It knows the note distribution and nothing
  about the conditioning, which is exactly the confound the next row removes.
- **unconditioned** is the shipped model given a zeroed grid and zeroed chroma, then
  scored against the real conditioning anyway. This is the ablation that makes the
  architecture claim falsifiable, and the claim survives it.
- **corpus** is the human-authored training MIDI scored on its own metrics — the ceiling.

## Does the validation loss mean anything? No

The README used to report a +0.022 gap between training and validation loss. That number
is not measuring generalisation, and the reason is the corpus redundancy described above.
Counted against the shipped data with `train_notes.py`'s own split seed:

| | bass | arp |
|---|---|---|
| validation examples | 637 | 518 |
| with another key of their own source file in train | **637 (100%)** | **518 (100%)** |
| with a **byte-identical** twin in train | 270 (42%) | 293 (57%) |
| genuinely held out | **0** | **0** |
| phrase families held out entirely | 0 | 0 |

Not one validation example was unseen. Roughly half were byte-for-byte copies of a
training row. A random split cannot hold anything out of a corpus where every phrase
appears ~94 times, so the gap it produces is a train-train gap and should be read as one.

`train_notes.py --split-by family` fixes this: it groups on the phrase identity recorded
in `training_data/manifests/`, holding out 7 of 45 bass phrases and 6 of 45 arp phrases
with zero overlap, asserted by `test_grouped_split_keeps_every_key_of_a_phrase_on_one_side`.
It is the default for new runs. The shipped checkpoints were trained under
`--split-by random`, and they have **not** been retrained — the models work, and a revised
loss number is not worth disturbing them for. So the honest statement is that this repo
has no measured generalisation gap, not that it has a good one.

Reproduce the leak counts with `PYTHONPATH=. python experiments/split-leak/measure.py`.

## Did DPO learn taste, or memorise ten clips? Unknown, leaning memorise

On the pairs it was trained on, bass preference accuracy climbs 23% → 40% → 95% → 100%
across base → ft1 → ft2 → ft3. That is the flattering chart and it measures absorption,
not taste: a log-probability of −6.9 across 65 tokens is about 90% per token on data the
model was fit to reproduce.

On the one edit never used in training, preference accuracy is **0% at every stage** —
base, ft1, ft2 and ft3 alike. The margin does not merely fail to become positive; it stays
around −90 to −120 nats throughout.

| | training pairs (n=10) | held out (n=1) |
|---|---|---|
| base | 23% | 0% |
| ft1 | 40% | 0% |
| ft2 | 95% | 0% |
| ft3 | **100%** | **0%** |

**With n=1 this is not a result.** One held-out clip cannot distinguish "learned nothing
transferable" from "unlucky draw", and it would be as wrong to report 0% as a finding as
to report the 100% alone. What it does establish is that the 100% column carries no
evidence about generalisation, which is the claim someone would otherwise read into it.

A real answer needs roughly 30 held-out pairs — enough that a binomial interval around the
accuracy excludes chance — collected the same way as the training pairs and never shown to
a fine-tune. Everything needed to collect them is in the UI's fine-tune tab; the missing
ingredient is edits, not code.

There is also a representation ceiling on this measurement: 2 of the 10 training pairs
tokenize identically to the clip they were edited from, because those edits were velocity
and sub-grid timing. The preference objective cannot see them at all. See
[Why this representation](#why-this-representation-and-not-remi-or-note-onnote-off).

# Retraining from scratch

The whole path is in the repository. Nothing here is needed to generate — only to
rebuild the models.

```bash
# 1. corpus of MIDI -> training examples  (reads training_data/corpus by default)
uv run --frozen python -m train.mine_corpus --role bass
uv run --frozen python -m train.mine_corpus --role arp

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
It prints the drop at every stage and writes `training_data/manifests/<role>.json`,
which records that funnel and maps each example's content digest back to the file it
came from — the multitrack corpus names the folder and calls the file `bass.mid`, so
without the manifest nothing downstream can say what an example is.

It also reports **distinct phrase families** alongside unique files, and the two do not
agree here: 354 bass files are 45 phrases. Mine an un-augmented corpus if you have the
choice, and group the validation split on `family` either way.

**Training time, measured.** A full 150-epoch run costs **7.2 minutes on MPS** and
**12.8 minutes on CPU** (median 2.89 s and 5.11 s per epoch including the validation
pass, M-series, torch 2.12). The README and the write-up used to quote these as if they
disagreed — "minutes on the CPU" against "about seven and a half minutes on an M-series
GPU" — when they were describing different devices and both were right.

The interesting part is that this is the **opposite** of the inference result above,
where CPU beats MPS by 8.3×, and the reversal has one cause. Training is teacher-forced:
a batch of 16 examples, all 65 positions at once, one kernel launch covering 1,040
positions of work. Generation is batch-1 and sequential: 64 launches covering one
position each. Same model, same device — the accelerator wins whenever there is enough
parallel work in a launch to pay for the launch, and loses when there is not.

Reproduce with `PYTHONPATH=. python experiments/bench/train_throughput.py`. It times the
real loop and **never saves a checkpoint**, so it cannot disturb the shipped weights.

The models are small enough that the binding constraint is corpus size, not compute — and
with 45 distinct phrases per role, that constraint binds hard.


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

The note models are **621,184 parameters** each: 65-token vocabulary, 3 layers, 128
dimensions, 4 heads. Both roles use the identical configuration; only the weights and
the mono-reduction rule (`lowest` for bass, `highest` for arp) differ.

A checkpoint file sums to 700,720 instead, and the discrepancy is worth one line because
it used to be quoted as three different numbers. 621,184 is the live architecture.
Adding the retired `cond_proj` — a `Linear(512, 128)` for a CLAP embedding nothing here
can produce, removed from the model but still carried as untouched init in all eight
shipped checkpoints — gives 686,848, which is where "0.687M" came from. Adding the three
68×68 causal masks, which are buffers rather than parameters, gives 700,720.
`test_parameter_count_is_the_number_the_docs_quote` pins all three.

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

**Device.** Sampling runs on the CPU by default, which is **8.3× faster than MPS** here.
Generation is 64 sequential single-token passes through a 621k-parameter model, and at
that size accelerator launch overhead costs more than the arithmetic saves:

| | CPU | MPS | ratio |
|---|---|---|---|
| one 4-bar candidate | **51.8 ms** | 431.4 ms | 8.3× |
| 8-bar track, best-of-10, both stems | **2.07 s** | 17.9 s | 8.7× |
| end to end — request to MIDI on disk | **2.11 s** | 18.1 s | 8.6× |

Medians over 30 runs (3 for the track and end-to-end rows), M-series, torch 2.12.
End-to-end includes drum-loop decoding, BPM detection, tempo alignment, the onset grid,
loading both checkpoints, and writing MIDI. Reproduce with
`PYTHONPATH=. python experiments/bench/latency.py`; the raw output is
`experiments/bench/latency.json`. `--device auto` selects an accelerator if one is
present.

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
| `corpus/` | 19,065 source `.mid` — the packs the examples are mined from |
| `manifests/` | `bass.json`, `arp.json` — the funnel, and every digest's source file |
| `bass_notes_midi/` | 4,248 mined bass `.npz` — 354 sources, 45 distinct phrases |
| `arp_notes_midi/` | 3,456 mined arp `.npz` — 288 sources, 45 distinct phrases |
| `dpo/` | 5 tracks of hand-edited MIDI preference pairs |
| `drum_loops/` | the loop those pairs were conditioned on |

Two things here are not tracked, for the same licensing reason at different strengths.
`corpus/` is purchased sample-pack MIDI verbatim; the licence covers using the packs, not
redistributing them. The mined `.npz` are lossy but musically recognizable derivatives of
it — a clone could reconstruct the rhythm and intervals of 45 basslines from them.

The manifests **are** tracked, and they are the part a reader needs: they name the source
file behind every example, which the `.npz` filenames cannot, since those keep only the
first 28 characters of a stem that is usually just `bass`. The DPO pairs are tracked too
— they are the only human-authored data in the project. `training_data/README.md`
documents how everything was produced and how to rebuild it.

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
| `mine_corpus.py` | MIDI corpus -> training examples, funnel, provenance manifest |
| `train_notes.py` | training examples -> base checkpoint; phrase-grouped split |
| **`ui/`** | |
| `app.py` | Streamlit front end: generate and fine-tune tabs |
| **`experiments/`** | every published number has its generator here |
| `baselines/run.py` | random / bigram / ablation / human-corpus reference points |
| `split-leak/measure.py` | how much of the validation set is already in train |
| `bench/latency.py` | generation latency, CPU vs MPS |
| `bench/train_throughput.py` | training time per device; writes no checkpoint |
| `figures/kl_drift.py` | `figures/kl-drift.png` from `kl_history.json` |
| `checkpoint-drift/` | where each `ftN` landed; preference, held out and not |
| **`tests/`** | |
| `test_inference.py` | tokenizer, conditioning, chords, checkpoints, corpus provenance |
| `test_finetune_dpo.py` | iterated DPO: distribution moves, KL stays bounded (slow) |
| **root** | |
| `pipeline.html` | standalone diagram |
| `LICENSE` | MIT for the code; see the note about the sample-pack MIDI |

## Test

```bash
uv run --frozen pytest              # fast suite, ~6s, 123 tests
uv run --frozen pytest -m slow      # iterated DPO, runs real training rounds
```

CI runs the fast suite on every push (`.github/workflows/tests.yml`) and additionally
asserts that no sample-pack MIDI and no model weights outside `checkpoints/` are tracked
— the two mistakes a careless `git add -A` would make in this repo.

## Reproducing the numbers

Every figure and measured claim has a generator under `experiments/`, which is tracked
for that reason. None of these retrain the shipped models.

```bash
PYTHONPATH=. uv run --frozen python experiments/baselines/run.py           # the baseline table
PYTHONPATH=. uv run --frozen python experiments/split-leak/measure.py      # the validation leak
PYTHONPATH=. uv run --frozen python experiments/bench/latency.py           # CPU vs MPS generation
PYTHONPATH=. uv run --frozen python experiments/bench/train_throughput.py  # training time, saves nothing
PYTHONPATH=. uv run --frozen python experiments/figures/kl_drift.py        # figures/kl-drift.png
```

See `experiments/README.md`.

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

### The taste loop

The distinctive thing in this project is that the loop closes: the model that generates
your next clip is the model your last edit changed. Nothing leaves the machine.

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
   1. generate      2. download      3. edit          4. upload    │
   ──────────       ───────────      ───────         ──────────    │
   kick grid   ──▶  stems/bass.mid ─▶ in a DAW ────▶ bass_edited   │
   + chroma         (REJECTED)       fix what        .mid          │
        │           the model         is wrong      (CHOSEN)       │
        │           produced                            │          │
        ▼                                               ▼          │
   HarmonicNoteGPT                              5. DPO on the pair │
   (checkpoint ftN)                             ──────────────────  │
        ▲                                       KL-penalised step  │
        │                                       against ftN as the │
        │                                       reference policy   │
        │                                               │          │
        └───────── 6. ftN+1 becomes the default ────────┘          │
                      for the next generation                      │
                                                                   │
                    └──────────────────────────────────────────────┘
```

Six steps, and the two that matter are the ones with a number attached:

1. **Generate** conditioned on a drum loop's kick pocket and a chord progression.
2. **Download** the stem. What the model produced is, by construction, the `rejected`
   side of the pair — you only edit what you want changed.
3. **Edit** it in Ableton. This is the only human-authored signal anywhere in the
   project; `training_data/dpo/` holds 20 such files.
4. **Upload** it back as `<role>_edited.mid` beside the original. `metadata.json` is what
   makes this work: the trainer rebuilds the exact conditioning the clip was generated
   under, so the pair is compared under identical circumstances rather than approximately.
5. **DPO** with an adaptively weighted KL penalty against the previous checkpoint. The
   weight is tuned to hold drift inside a target budget, so taste moves without the
   distribution collapsing — `test_iterated_dpo_holds_the_kl_budget` asserts this across
   real training rounds, not on a mock.
6. **`ftN+1` becomes the default.** `latest_ckpt()` picks the highest `ftN` on disk, so
   the next generation is on-policy: your next edit targets the taste that is *left*,
   not the mistake you already corrected.

The measured behaviour of steps 5–6 is in `experiments/checkpoint-drift/`, including the
honest version — on its training pairs preference accuracy reaches 100%, and on the one
edit it never saw it is 0%. With n=1 that is not a result; see **Evaluation** above for
what a real one needs.

There is no audio playback and no piano roll — the stems go into a DAW, which does both
better.

`.streamlit/config.toml` turns off Streamlit's usage telemetry, which otherwise posts to
`webhooks.fivetran.com` and `data.streamlit.io` on every session. With it in place the UI
makes no external requests.
