"""Scoring a generated section against the conditioning it was given."""
from __future__ import annotations

import numpy as np

from utils.config import NOTE_BARS, NOTE_MIDI_LO, NOTE_PITCH0


def fit_and_lock(tokens, chroma, grid):
    """Validation: chord-tone fit + kick-lock for one 4-bar section."""
    fit = tot = 0
    for b in range(NOTE_BARS):
        ch = chroma[b * 16]
        for t in tokens[b * 16:(b + 1) * 16]:
            if t >= NOTE_PITCH0:
                tot += 1; fit += ch[(NOTE_MIDI_LO + (t - NOTE_PITCH0)) % 12] > 0
    on = np.array([1.0 if t >= NOTE_PITCH0 else 0.0 for t in tokens])
    def z(x):
        x = x - x.mean(); s = x.std(); return x / s if s > 1e-8 else x
    lock = float(np.dot(z(on), z(grid)) / len(grid)) if on.sum() else 0.0
    return (fit / tot if tot else 0.0), lock
