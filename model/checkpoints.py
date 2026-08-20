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


def load_note_model(ckpt: Path, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = HarmonicNoteGPT(ck["config"]).to(device).eval()
    m.load_state_dict(ck["model"])
    return m
