"""Chord progression parsing and per-step chroma conditioning."""
from __future__ import annotations

import numpy as np

PC = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = {
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "m7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "7": (0, 4, 7, 10),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus4": (0, 5, 7),
    "sus2": (0, 2, 7),
    "min": (0, 3, 7),
    "m": (0, 3, 7),
    "maj": (0, 4, 7),
    "": (0, 4, 7),
}


def chord_chroma(root_pc: int, quality: str = "min") -> np.ndarray:
    vector = np.zeros(12, np.float32)
    for interval in QUALITIES[quality]:
        vector[(root_pc + interval) % 12] = 1.0
    return vector


def parse_chord(token: str) -> tuple[int, str]:
    token = token.strip()
    root = token[:2] if len(token) > 1 and token[1] in "#b" else token[:1]
    root_pc = PC.get(root.upper())
    if root_pc is None:
        raise ValueError(f"bad chord root: {token!r}")
    quality = token[len(root):].strip().lower()
    if quality == "-":
        quality = "min"
    if quality not in QUALITIES:
        raise ValueError(f"unknown chord quality {quality!r} in {token!r}")
    return root_pc, quality


def parse_progression(progression: str) -> list[tuple[int, str]]:
    for separator in ("–", "—", ",", "|"):
        progression = progression.replace(separator, "-")
    chords = [parse_chord(token) for token in progression.split("-") if token.strip()]
    if not chords:
        raise ValueError("progression must contain at least one chord")
    return chords


def progression_to_track(progression: str, n_steps: int) -> np.ndarray:
    chords = parse_progression(progression)
    track = np.zeros((n_steps, 12), np.float32)
    segment = n_steps / len(chords)
    for index, (root, quality) in enumerate(chords):
        lo = int(round(index * segment))
        hi = int(round((index + 1) * segment))
        track[lo:hi] = chord_chroma(root, quality)
    return track
