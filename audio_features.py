"""Audio features used to condition note generation."""
from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from config import (
    GRID_FRAMES,
    GRID_STEPS,
    FRAME_RATE,
    KICK_BAND_HZ,
    SAMPLE_RATE,
)


def _fit_len(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return values[:length]
    pad = values[-1] if len(values) else 0.0
    return np.concatenate([values, np.full(length - len(values), pad, values.dtype)])


def onset_grid(waveform: np.ndarray) -> np.ndarray:
    """Extract the normalized low-frequency kick-pocket grid used in training."""
    sos = butter(4, KICK_BAND_HZ, btype="low", fs=SAMPLE_RATE, output="sos")
    low = sosfiltfilt(sos, waveform.astype(np.float64))
    hop = int(round(SAMPLE_RATE / FRAME_RATE))
    envelope = librosa.onset.onset_strength(
        y=low.astype(np.float32),
        sr=SAMPLE_RATE,
        hop_length=hop,
    )
    envelope = _fit_len(envelope, GRID_FRAMES)
    bins = np.array_split(envelope, GRID_STEPS)
    steps = [chunk.max() if chunk.size else 0.0 for chunk in bins]
    grid = np.concatenate([
        np.full(len(chunk), value, np.float32)
        for chunk, value in zip(bins, steps)
    ])
    grid = _fit_len(grid, GRID_FRAMES)
    peak = grid.max()
    return (grid / peak).astype(np.float32) if peak > 0 else grid.astype(np.float32)
