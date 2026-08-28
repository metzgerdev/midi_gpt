# experiments

Every published number and figure is produced by a script in here. That is the point of
the directory: `.gitignore` used to exclude all of `experiments/`, which left `figures/`
as a set of screenshots with no generator, and left three README claims — the parameter
count, the training time, the CPU-beats-GPU latency — with nothing behind them at all.

The `.py`, `.md` and measured `.json` are tracked. **`.pt` files are not**, and must not
be: `latest_ckpt()` globs `CKPT_DIR` and takes the highest `ftN` on disk, so an
experimental checkpoint that reaches `checkpoints/` silently becomes the default for
generation *and* the policy start for the next DPO round. Give any training run an
explicit `--out` under here.

Nothing in this directory retrains the shipped models. The two that touch a training loop
say so in their own docstring and neither writes a checkpoint.

| directory | what it answers | trains? |
|---|---|---|
| `baselines/` | Is chord_fit 0.937 good? Floor, bigram, ablation, human ceiling. | no |
| `bench/latency.py` | Is CPU really faster than MPS here, and by how much? | no |
| `bench/train_throughput.py` | How long does a full training run take, per device? | times the loop, **saves nothing** |
| `split-leak/` | How much of the validation set is already in train? | no |
| `checkpoint-drift/` | Where each `ftN` landed: corpus loss, KL, weight drift, preference. | no |
| `figures/kl_drift.py` | Does the KL penalty hold drift across iterated DPO rounds? | no |
| `loss-curves/` | The training curves in the write-up. | yes — use `--out` |
| `dpo_rounds/` | Weights from the iterated-DPO experiment. Untracked. | — |

## Running them

All of them expect the repo root on the path:

```bash
PYTHONPATH=. uv run --frozen python experiments/baselines/run.py
PYTHONPATH=. uv run --frozen python experiments/bench/latency.py
PYTHONPATH=. uv run --frozen python experiments/bench/train_throughput.py
PYTHONPATH=. uv run --frozen python experiments/split-leak/measure.py
PYTHONPATH=. uv run --frozen python experiments/figures/kl_drift.py
```

`baselines/` and `split-leak/` need the mined `.npz`, which are gitignored; `latency.py`
and `kl_drift.py` need only what the repo ships. Anything that cannot find its inputs
says so and exits rather than producing an empty result.
