**Midi GPT**


A small language model that generates MIDI, trained on a curated data set, and fine tuned further with Direct Preference Optimization.


**Motivation**


I produce electronic music, and need a tool to help me get past creative blocks. A short musical pattern is often enough to build into something bigger. Existing MIDI generators produce outputs that are not in my style or voice. Midi GPT is aimed at a narrow slice of EDM and can be further fine tuned to a user's preference. 


**Data Preparation**


I trained two models, one for bass and one for melody. I curated a collection of MIDI patterns from House and UK Garage.


Each training example is one 4-bar phrase, reduced to a single voice and written as one token per sixteenth note indicating a pitch, a rest, or a sustain. A distilled representation of music pinned to rhythm and pitch.


Each chunk carries two additional signals derived from itself: **grid** (its own note-ons) and **chord** (inferred per bar by matching a pitch-class histogram against chord templates). The corpus is thus self-supervised.


Every chunk is transposed through all twelve keys - 354 unique bass phrases become 4,248 training examples, and 288 melody phrases become 3,456.  This data augmentation also removes the correlation between rhythm being tied to a certain pitch, so that the model doesn't overfit rhythm with pitch.


**Token Vocabulary**


The vocabulary is 65 symbols: REST = 0, SUSTAIN = 1, 61 pitches (MIDI 24–84, C1 through C6), then BOS - (Beginning of Sequence) = 63 and EOS (End of Sequence) = 64.  The tokenization is a direct mapping of the symbol to token id.
![The 65-token vocabulary laid over a keyboard: REST and SUSTAIN, 61 pitch ids from C1 to C6, then BOS and EOS](figures/vocab-token-ids.png)


The model has a simple job at each step — output a pitch, a rest, or a sustain.  The sequence [2, 1, 1, 1] is note C1 sustained for a quarter note (each step is a 1/16th note). The resulting midi is edited in a digital audio workstation. 

**Embedding**

Every token id becomes a 128-d vector by indexing a row of `nn.Embedding(65, 128)` — row 14 is C2. The lookup is exactly `one_hot(14) @ W`.

![The nn.Embedding(65, 128) table, one row per token, with row 14 (C2) highlighted](figures/embedding-table.png)
![One-hot(14) times W equals row 14 of the table](figures/embedding-lookup.png)

Position indexes its own table, similar to the token id look up. Grid and chord are projected rather than looked up, since their inputs are values, not ids. A chord is a 12-slot chroma, one slot per pitch class:

![A 12-slot chroma vector for A minor, with C, E and A set to 1 and the rest 0](figures/chord-chroma.png)

`nn.Linear(12, 128)` maps it up, which for A minor is just three columns and a bias:

chord_proj(A minor) = W[:,C] + W[:,E] + W[:,A] + b

![The three active pitch-class rows of the linear layer summing into one 128-d vector](figures/chord-projection.png)

All four land in the same 128-d space and are summed:

x = tok_emb + pos_emb + grid_proj + chord_proj

| signal | shape in | how it becomes 128-d |
| --- | --- | --- |
| token | 1 id (0–64) | `nn.Embedding(65, 128)` — the id indexes a row |
| position | 1 index (0–67) | `nn.Embedding(68, 128)` — the step indexes a row |
| grid | 1 scalar (0 or 1) | `nn.Linear(1, 128)` — one weight vector scaled by the onset |
| chord | 12-d chroma | `nn.Linear(12, 128)` — a learned map from pitch classes |

Grid and chord user provided signals to condition the output. In training both come from the phrase itself — the stem's own note-ons, and the chroma inferred per bar. At inference the grid comes from the kick of a chosen drum groove, four-on-the-floor house or two-step UK Garage, and the chord from your key and progression.

**Transformer**

The backbone is a small GPT-2. At roughly 620k parameters, the model can easily run on a laptop.

| key | value | what it is |
| --- | --- | --- |
| `n_layers` | 3 | transformer blocks |
| `n_heads` | 4 | attention heads per block, 32 dimensions each |
| `emb_dim` | 128 | model width — every signal is projected to this |
| feed-forward | 512 | hidden width inside each block, 4 × `emb_dim` |
| `context_length` | 68 | BOS + 64 steps + EOS, with room to spare |
| `vocab_size` | 65 | REST, SUSTAIN, 61 pitches, BOS, EOS |
| `drop_rate` | 0.1 | |
| `qkv_bias` | False | |


**Training**

Training is teacher-forced.  A causal mask prevents look ahead attention. AdamW, learning rate 5e-4, weight decay 0.05, batch size 16, 150 epochs, 15% held out for validation, best validation loss kept.

Training took about seven and a half minutes on
an M-series GPU.

Most of the drop happens in the first fifteen epochs — cross-entropy falls from 2.0 to about
0.18 — and from roughly epoch 40 the run is grinding out small improvements. Validation
bottoms at 0.1243 on epoch 131 and drifts up slightly afterwards.

![Training and validation cross-entropy over 150 epochs of the bass model, shown whole and zoomed from epoch 20, with a generalisation gap of +0.022 and best validation 0.1243 at epoch 131](figures/loss-curves-bass.png)

 

**SFT**


From the base checkpoint, I generated 10 outputs, and edited them to my preference.  These edits were transposed through a range of 12 semitones and concatented with the existing corpus into a shuffled loader for SFT.  This mixture, along with a low learning rate (1e-4), and only 8 epochs, limits drift from the base model.  As seen below, the goal is to shift the distribution towards the preference, but not destroy the base model.


![Mean log P(clip) across base, ft1, ft2 and ft3: the chosen line rises from -133 to -7, the rejected line lags](figures/sft-preference-shift.png)


**DPO Fine Tune**


DPO fine-tunes further toward the user's preference. It optimises an implicit reward — how much more likely the policy generates a clip than the frozen anchor does — and widens the gap between that reward for the clip you kept and the clip you replaced, while a KL term holds the policy near the anchor. 


sequence_logprob sums the per-step log probabilities of one exact clip — how likely the model was to produce that precise sequence, under the grid and chord it was conditioned on (g and c below). The policy is the model being trained; it starts as the checkpoint that generated the clip. The anchor is frozen, and by default it is the untuned base model.  Drift is measured from the base checkpoint instead of compounding across rounds. 


```python
policy_chosen   = sequence_logprob(policy, xc, yc, g, c)
policy_rejected = sequence_logprob(policy, xr, yr, g, c)
anchor_chosen   = sequence_logprob(anchor, xc, yc, g, c)   # frozen
anchor_rejected = sequence_logprob(anchor, xr, yr, g, c)   # frozen


margin = beta * ((policy_chosen - anchor_chosen)
                - (policy_rejected - anchor_rejected))
loss   = -F.logsigmoid(margin).mean() + lam * kl_to_anchor(policy, contexts)
```
![Implicit reward over 30 DPO rounds on 24 held-out pairs: r(chosen) climbs to +5.17 while r(rejected) falls to -2.36, a margin of +7.53](figures/dpo-reward-margin.png)















