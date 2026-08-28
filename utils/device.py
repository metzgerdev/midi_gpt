"""Device selection for sampling and training."""
from __future__ import annotations

import torch


def pick_device(name: str | None = None) -> torch.device:
    """CPU by default — measured 8.3x faster than MPS for this model.

    Generation is 64 sequential single-token passes through a 0.62M-parameter model.
    The tensors are small enough that accelerator launch and sync overhead dominates
    the arithmetic: 431 ms per candidate on MPS against 51.8 ms on CPU (medians of 30
    runs, M-series, torch 2.12), so an 8-bar best-of-10 run costs 18.1 s end to end on
    MPS and 2.1 s on CPU.

    Reproduce with `PYTHONPATH=. python experiments/bench/latency.py`; the numbers above
    and in the README come from experiments/bench/latency.json. They were previously
    quoted as 632/60 ms and 25/2.4 s, which had no benchmark behind them.

    The accelerator branch is kept rather than deleted because the trade flips if the
    sampler is ever batched or the model grows. `--device auto` restores the old
    behavior. Note that a seed does not carry across devices: MPS and CPU diverge
    within a few steps, so runs are reproducible on one device, not between two.
    """
    if name in ("cpu", "mps", "cuda"):
        return torch.device(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")
