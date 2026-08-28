# Checkpoint drift measurement

No SFT or DPO run in this repo ever logged its training history, so there is no
training curve for the shipped `ftN` checkpoints. These scripts measure the
checkpoints as they exist instead — endpoint measurements, not a path.

    PYTHONPATH=. python experiments/checkpoint-drift/measure_checkpoints.py   # -> measurements.json
    PYTHONPATH=. python experiments/checkpoint-drift/measure_quality_se.py    # -> quality_samples_n200.json
    PYTHONPATH=. python experiments/checkpoint-drift/build_chart_data.py
    PYTHONPATH=. python experiments/checkpoint-drift/render_chart.py          # -> checkpoint-drift.html

Deterministic: corpus loss, KL from base, per-layer weight drift.
Sampled (carry noise): chord fit, kick lock — 200 clips/stage, reported with SE.

Lineage read from checkpoint metadata: base -> ft1 (SFT, 4 edits, superseded);
base -> ft2 (SFT, 10 edits) -> ft3 (DPO, beta=0.1).

## Preference shift (pref.py / heldout.py / pref_chart.py)

Log-probability each checkpoint assigns to the kept clip vs the one it replaced,
using the repo's own `preference_pairs` and `sequence_logprob`.

    PYTHONPATH=. python experiments/checkpoint-drift/pref.py       # training pairs (training_data/dpo)
    PYTHONPATH=. python experiments/checkpoint-drift/heldout.py    # the one edit not used in training
    PYTHONPATH=. python experiments/checkpoint-drift/pref_chart.py # -> preference-shift.html

On its training pairs, preference accuracy goes 23% -> 40% -> 95% -> 100% (bass).
On the held-out edit it is 0% at every stage.
