"""Finding and loading trained note checkpoints."""
from __future__ import annotations

import re
from pathlib import Path

import torch

from utils.config import CKPT_DIR
from model.note_model import HarmonicNoteGPT


def latest_ckpt(role: str) -> Path:
    """Newest fine-tuned <role>_notes_gpt_ftN.pt (highest N), or the base checkpoint if none.

    On-policy default: generate from the model that already absorbed prior edits, so new
    hand-edits target the residual taste instead of re-teaching what SFT already learned."""
    base = CKPT_DIR / f"{role}_notes_gpt.pt"
    fts = [(int(m.group(1)), p) for p in CKPT_DIR.glob(f"{role}_notes_gpt_ft*.pt")
           if (m := re.search(r"_ft(\d+)$", p.stem))]
    return max(fts)[1] if fts else base


# Keys for layers the model no longer has. `cond_proj` was a Linear(512, 128) for a CLAP
# clip embedding nothing in this repo could produce; every call passed cond=None, so it
# never entered the graph and never took a gradient. All eight shipped checkpoints were
# written while it existed and still carry its 65,664 untouched init weights, so every
# load has to drop them or `load_state_dict` raises on the unexpected keys.
RETIRED_KEYS = ("cond_proj.weight", "cond_proj.bias")


def load_checkpoint(ckpt: Path, device) -> dict:
    """Load a checkpoint dict, dropping keys for layers that have since been removed.

    `pop(..., None)` is idempotent, so this reads both the checkpoints written before
    the CLAP path came out and any written after. Every load site goes through here —
    they had drifted into four hand-rolled `torch.load` + `load_state_dict` pairs, and
    a retired key missed at any one of them is a `RuntimeError` at that call site only.
    """
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    for key in RETIRED_KEYS:
        saved["model"].pop(key, None)
    return saved


def model_from_checkpoint(saved: dict, device, config: dict | None = None):
    """Build a HarmonicNoteGPT from an already-loaded checkpoint dict.

    Split out from `load_note_model` for callers that also want the metadata beside the
    model and should not read the file twice to get it. The load is strict on purpose:
    the retired keys are gone by the time `load_checkpoint` returns, so anything else
    unexpected is a real mismatch and should say so.
    """
    m = HarmonicNoteGPT(config or saved["config"]).to(device).eval()
    m.load_state_dict(saved["model"], strict=True)
    return m


def load_note_model(ckpt: Path, device, config: dict | None = None):
    """Build a HarmonicNoteGPT from a checkpoint path and load its weights.

    Returns the model in eval mode; callers that train it set `.train()` themselves.
    `config` overrides the checkpoint's own — pass it to assert a known architecture
    rather than trusting whatever shape the file names.
    """
    return model_from_checkpoint(load_checkpoint(ckpt, device), device, config)
