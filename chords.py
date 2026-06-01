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


# Diatonic scales, as semitone offsets from the key root. Natural minor rather than
# harmonic — a raised 7th would put a major dominant in the key, which is exactly the
# color the dark UKG mood template avoids.
SCALES = {
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "major": (0, 2, 4, 5, 7, 9, 11),
}

# Candidate recolorings for a chord, by its plain triad quality. Which of these
# survive is decided by the key, not by taste: see `color_progression`.
COLORS = {
    "m": ("m", "m7", "sus4", "sus2"),
    "": ("", "maj7", "7", "sus4", "sus2"),
}


def chord_chroma(root_pc: int, quality: str = "min") -> np.ndarray:
    vector = np.zeros(12, np.float32)
    for interval in QUALITIES[quality]:
        vector[(root_pc + interval) % 12] = 1.0
    return vector


# Chord shapes the corpus miner matches against. Order matters: when a bar supplies too
# few pitch classes to separate the candidates, several templates tie and the first wins.
INFER_TEMPLATES = [(root, quality)
                   for root in range(12)
                   for quality in ("min", "maj", "min7", "7", "dim", "sus4")]


def infer_chord(pc_weight: np.ndarray) -> np.ndarray:
    """Weighted pitch-class histogram -> the chroma of the best-matching chord.

    Score is the mass landing on chord tones minus half the mass landing off them.
    """
    if pc_weight.sum() <= 0:
        return np.zeros(12, np.float32)
    weights = pc_weight / pc_weight.sum()
    best, best_score = None, -1e9
    for root, quality in INFER_TEMPLATES:
        chroma = chord_chroma(root, quality)
        score = float((weights * chroma).sum() - 0.5 * (weights * (1 - chroma)).sum())
        if score > best_score:
            best, best_score = chroma, score
    return best


def infer_chord_track(step_pitch: np.ndarray, n_bars: int, steps_per_bar: int) -> np.ndarray:
    """Per-step MIDI pitch (-1 = rest) -> per-step chroma, one chord held per bar.

    The self-supervised harmony label. At training time the chord is read back out of
    the stem's own notes; at inference it comes from the progression the user asked for.
    Same 12-dimensional interface either way, which is what lets the model be trained
    without annotation and still be steered by a typed chord symbol.
    """
    track = np.zeros((n_bars * steps_per_bar, 12), np.float32)
    for bar in range(n_bars):
        segment = step_pitch[bar * steps_per_bar:(bar + 1) * steps_per_bar]
        histogram = np.zeros(12, np.float32)
        for pitch in segment:
            if pitch >= 0:
                histogram[int(pitch) % 12] += 1.0
        track[bar * steps_per_bar:(bar + 1) * steps_per_bar] = infer_chord(histogram)
    return track


def scale_pcs(key_root: int, mode: str) -> set[int]:
    """The pitch classes of a key, e.g. A minor -> {A B C D E F G}."""
    return {(key_root + step) % 12 for step in SCALES[mode]}


def in_key(root_pc: int, quality: str, scale: set[int]) -> bool:
    """True when every tone of the chord is diatonic to the key."""
    return set(np.flatnonzero(chord_chroma(root_pc, quality)).tolist()) <= scale


def color_progression(progression: str, key_root: int, mode: str, rng) -> str:
    """Recolor each chord with a 7th or suspension that stays in the key.

    Harmonic function is untouched — the roots and their major/minor quality come from
    the mood template and stay put. Only the color above the triad varies, and a
    candidate is kept only if all of its tones are diatonic, so correctness falls out
    of the key rather than out of a hand-written table. In A minor that admits Fmaj7
    on the VI and G7 on the VII while rejecting F7 (Eb) and Gmaj7 (F#) automatically.

    `rng` is a seeded random.Random, so a run's chords are reproducible from its seed.
    """
    scale = scale_pcs(key_root, mode)
    out = []
    for root_pc, quality in parse_progression(progression):
        base = "m" if quality in ("m", "min") else ""
        options = [q for q in COLORS.get(base, (base,)) if in_key(root_pc, q, scale)]
        if base not in options:                    # the plain triad may leave the key
            options.append(base)                   # (borrowed chords); keep it reachable
        out.append(NAMES[root_pc] + rng.choice(options))
    return "-".join(out)


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
