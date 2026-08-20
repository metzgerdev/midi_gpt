"""Reading what is on disk: drum loops, checkpoints, and past runs."""
from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

from utils.config import CKPT_DIR, DRUM_DIR, OUTPUT_DIR


@dataclass
class Run:
    """One generated track, as recorded by make_track."""

    path: Path
    meta: dict

    @property
    def name(self) -> str:
        return self.path.name

    def stem(self, role: str) -> Path:
        return self.path / "stems" / f"{role}.mid"

    def edit(self, role: str) -> Path:
        return self.path / "stems" / f"{role}_edited.mid"

    def roles(self) -> list[str]:
        return [r for r in ("bass", "arp") if self.stem(r).exists()]

    def edited_roles(self) -> list[str]:
        return [r for r in ("bass", "arp") if self.edit(r).exists()]


def drum_loops() -> list[Path]:
    return sorted(DRUM_DIR.glob("*.wav"))


def checkpoints(role: str) -> list[Path]:
    """Base first, then the ftN chain in order."""
    base = CKPT_DIR / f"{role}_notes_gpt.pt"
    tuned = sorted(
        (p for p in CKPT_DIR.glob(f"{role}_notes_gpt_ft*.pt")),
        key=lambda p: int(re.search(r"_ft(\d+)$", p.stem).group(1)),
    )
    return ([base] if base.exists() else []) + tuned


def checkpoint_meta(path: Path) -> dict:
    """The recipe a checkpoint records. Older ones carry almost nothing."""
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in saved.items() if k not in ("model", "config")}


def runs(root: Path = OUTPUT_DIR) -> list[Run]:
    """Every run with readable metadata, newest first."""
    found = []
    for meta_path in root.glob("*/metadata.json"):
        try:
            found.append(Run(meta_path.parent, json.loads(meta_path.read_text())))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda r: r.meta.get("created", ""), reverse=True)


def capture(fn, *args, **kwargs):
    """Run something that prints, and hand back (result, what it printed).

    make_track and finetune_dpo report through stdout. The UI wants that text on the
    page rather than in the terminal Streamlit was launched from.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            result = fn(*args, **kwargs)
        except SystemExit as stop:                  # both scripts exit on bad input
            return None, f"{buffer.getvalue()}\n{stop}"
    return result, buffer.getvalue()
