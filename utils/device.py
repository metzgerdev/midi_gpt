"""Device selection for sampling and training."""
from __future__ import annotations

import torch


def pick_device(name: str | None = None) -> torch.device:
    """CPU by default — measured ~10x faster than MPS for this model.

    Generation is 64 sequential single-token passes through a 0.70M-parameter model.
    The tensors are small enough that accelerator launch and sync overhead dominates
    the arithmetic: 632 ms per candidate on MPS against 60 ms on CPU, so an 8-bar run
    costs ~25 s on GPU and ~2.4 s on CPU.

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
