# MIDI GPT

MIDI GPT is a local, conditioned music-generation system for house and UK garage.
It turns a text request, a drum loop, and a chord progression into bass and arp MIDI
stems that can be opened and edited in a DAW.

The project combines:

- Two GPT-style note models trained from scratch.
- Audio feature extraction for kick-pattern conditioning.
- Chord-aware symbolic generation.
- Best-of-N candidate selection with measurable constraints.
- A Streamlit interface for generation and fine-tuning.
- A human-in-the-loop DPO workflow that learns from DAW edits.
- Reproducible experiments, provenance manifests, and automated tests.

The system runs entirely on the local machine. It generates symbolic MIDI; audio
playback, sound selection, and mixing remain in the DAW.

## Results

The shipped fine-tuned models contain 621,184 parameters each. On an Apple Silicon
machine, an 8-bar generation run takes about 2.1 seconds end to end on CPU, including
drum-loop analysis, model loading, candidate selection, and MIDI writing.

On 240 conditioning cases using best-of-10 selection:

| Role | Chord-tone fit | Kick lock |
|---|---:|---:|
| Bass | 0.937 ± 0.010 | 0.886 ± 0.015 |
| Arp | 0.938 ± 0.010 | 0.810 ± 0.038 |

These are constraint-satisfaction metrics, not measures of musical quality. The
candidate-selection rule uses the same metrics, so the results should be read as
evidence that conditioning works—not as a claim that the model writes better music
than a producer.

## How it works

```text
text request ──┐
               ├─ chord progression ─────────────┐
drum loop ─────┘                                  │
                  kick onset grid ───────────────┤
                                                   ▼
                                  conditioned bass GPT ── bass.mid
                                  conditioned arp GPT  ── arp.mid
                                                   │
                                      score candidates and save MIDI
```

Each model generates one token per sixteenth-note step across a four-bar section.
Tokens represent a pitch, REST, or SUSTAIN. Notes are reconstructed from pitch and
sustain tokens, so the model is monophonic and does not represent velocity, polyphony,
or sub-sixteenth timing.

At every step, the model combines:

- The previous note token.
- A positional embedding.
- A kick-grid feature extracted from the drum loop.
- A 12-dimensional chord-chroma feature.

The model uses three transformer layers, 128-dimensional embeddings, and four attention
heads. Bass uses the lowest voice from a stem; arp uses the highest voice.

### Kick and harmony conditioning

The drum loop is low-pass filtered and reduced to a kick-pocket onset grid. The first
two bars are represented as 32 sixteenth-note values and tiled across each four-bar
section. The audio itself is never written to the output.

The requested key, mood, and progression become a per-step pitch-class mask. Chords can
also be colored with diatonic sevenths and suspensions.

### Candidate selection

The sampler generates several candidates for each section and selects the strongest
one using:

```text
2 × chord_fit + kick_lock + 0.7 × density_sanity
```

`chord_fit` is the proportion of generated note onsets landing on chord tones.
`kick_lock` is the correlation between generated note onsets and the kick grid.
Neither metric evaluates sound quality or producer intent.

## Human-in-the-loop fine-tuning

The main product loop is:

1. Generate a MIDI stem.
2. Edit it in a DAW.
3. Upload the edited stem beside the original.
4. Train with DPO, treating the edit as chosen and the original as rejected.
5. Use the resulting checkpoint for the next generation.

Each run stores its conditioning metadata, including the progression, drum loop, BPM,
seed, and model settings. The DPO trainer rebuilds the exact conditioning used for the
original generation, so both sides of a preference pair are compared under the same
conditions.

The DPO implementation uses a fixed anchor checkpoint and an adaptive KL penalty to
limit distribution drift. SFT is also available when direct training on edited MIDI is
preferred.

```bash
# Generate a track
uv run --frozen python -m inference.make_track \
  --request "8 bar UKG track in A minor, dark" \
  --seed 7

# Preview preference pairs without training
uv run --frozen python -m train.finetune_dpo --dry-run

# Fine-tune on edits
uv run --frozen python -m train.finetune_dpo

# Launch the local UI
uv run --frozen streamlit run ui/app.py
```

Generated runs are written to:

```text
output/<timestamp>_<key>_<mood>_<bars>bar[_<bpm>bpm][_ftN]/
├── stems/bass.mid
├── stems/arp.mid
└── metadata.json
```

## Training

The base models are trained directly on mined MIDI examples with next-token
cross-entropy. Onset tokens are weighted more heavily than sustain tokens so the model
does not learn the trivial solution of producing mostly rests and holds.

The source collection contains house and UK garage sample-pack MIDI. The training set is
organized around 45 four-bar patterns for each role, with every pattern expanded across
12 keys:

| Role | Patterns | Keys per pattern | Expanded patterns |
|---|---:|---:|---:|
| Bass | 45 | 12 | 540 |
| Melody / arp | 45 | 12 | 540 |

This is a deliberately focused corpus: one genre, two musical roles, and a limited set
of patterns. Transposing each pattern teaches the model to follow the chord conditioning
rather than memorize absolute pitches.


The source MIDI is purchased sample-pack material and is not redistributed. The code
is MIT licensed; see `training_data/README.md` for corpus provenance and licensing
details.

## Evaluation and limitations

The repository includes scripts under `experiments/` for baselines, latency, training
throughput, validation-split analysis, checkpoint drift, and preference shift.

The evaluation has three important limitations:

- The corpus contains only 45 underlying patterns per role, so it is intentionally narrow
  in both genre and musical vocabulary.
- The shipped models used a random validation split in which every validation example
  has a related key of the same source phrase in training. That split does not measure
  generalization. New training runs use phrase-grouped splits.
- DPO was trained on a small set of hand-edited role pairs: 10 bass pairs and 6 arp
  pairs. Two pairs tokenize identically because their edits changed unsupported MIDI
  properties. The current results do not establish that learned preferences generalize
  to new edits.

These limitations are documented and the corresponding measurements are reproducible;
they are part of the project’s engineering scope rather than claims hidden behind a
single validation score.

## Setup

Requirements:

- macOS
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- The shipped checkpoints in `checkpoints/`
- At least one drum loop in `drum_samples/`

Install dependencies:

```bash
uv sync --frozen
```

Run the fast test suite:

```bash
uv run --frozen pytest
```

The slow suite runs real DPO training rounds:

```bash
uv run --frozen pytest -m slow
```

## Repository structure

| Directory | Responsibility |
|---|---|
| `inference/` | CLI and notebook generation pipeline |
| `model/` | GPT backbone, conditioned note model, checkpoint loading |
| `utils/` | Chords, MIDI conversion, audio features, scoring, device selection |
| `train/` | Corpus mining, base training, SFT, and DPO |
| `ui/` | Streamlit generation and fine-tuning interface |
| `experiments/` | Reproducible benchmark and evaluation scripts |
| `tests/` | Inference, training, provenance, and UI tests |
| `training_data/` | Mined examples, manifests, and human-edited preference pairs |

Published figures and metrics have generators under `experiments/`. The repository’s
CI runs the fast tests and checks that licensed source MIDI and unintended model files
are not committed.
