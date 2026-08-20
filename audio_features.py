"""Audio features used to condition note generation."""
from __future__ import annotations

import re
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from config import (
    GRID_FRAMES,
    GRID_REF_BPM,
    GRID_STEPS,
    FRAME_RATE,
    KICK_BAND_HZ,
    SAMPLE_RATE,
)


def detect_bpm(path: Path) -> float | None:
    """Read a tempo out of the filename: 'house_drums_loop_127bpm' -> 127.

    Filename rather than beat tracking, which is unreliable on two bars of drums and
    would fail silently. Returns None when there is nothing to go on, and the caller
    then assumes the loop is already at GRID_REF_BPM.
    """
    stem = Path(path).stem.lower()
    m = re.search(r"(\d{2,3})\s*bpm", stem) or re.search(r"(?<!\d)(\d{2,3})(?!\d)", stem)
    if m and 60 <= int(m.group(1)) <= 200:
        return float(m.group(1))
    return None


def align_to_grid_tempo(audio: np.ndarray, loop_bpm: float) -> np.ndarray:
    """Time-stretch a loop so two of its bars fill the fixed analysis window.

    `onset_grid` always reads the first GRID_FRAMES frames — 3.693 s, which is exactly
    two bars at GRID_REF_BPM — and splits that into 32 sixteenth bins. A loop at any other
    tempo has bars of a different length, so its onsets land in the wrong bins and the
    error compounds across the bar. On a 127 BPM house loop that misplaced 9 of 32
    steps, including the downbeat kick.

    Stretching only aligns the analysis; the grid is tempo-agnostic in step space, and
    the output tempo is set separately by --bpm.
    """
    if abs(loop_bpm - GRID_REF_BPM) < 0.01:
        return audio
    return librosa.effects.time_stretch(audio, rate=GRID_REF_BPM / loop_bpm)


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
