"""Conversion between note tokens, MIDI events, and MIDI files."""
from __future__ import annotations

from pathlib import Path

import mido
import numpy as np

from config import (
    GRID_REF_BPM, NOTE_MIDI_HI, NOTE_MIDI_LO, NOTE_PITCH0, NOTE_REST, NOTE_SUSTAIN,
    NOTE_VOCAB,
)


def midi_to_step_grid(midi_file: mido.MidiFile, mono: str = "lowest"):
    """MIDI -> per-step sounding pitch (-1 = rest), reduced to a single voice.

    The inverse of `write_midi`, and the same reduction the training corpus was built
    with: `lowest` keeps the fundamental of a stacked bass, `highest` keeps the top
    line of a chord. Fine-tuning has to reproduce it exactly, or an edited clip would
    tokenize differently from the original it is paired against.
    """
    events, active, tick = [], {}, 0
    for message in mido.merge_tracks(midi_file.tracks):
        tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault(message.note, []).append(tick)
        elif message.type == "note_off" or (message.type == "note_on" and not message.velocity):
            if active.get(message.note):
                events.append((message.note, active[message.note].pop(0), tick))
    if not events:
        return None

    step_ticks = midi_file.ticks_per_beat / 4
    n_steps = int(np.ceil(max(end for _, _, end in events) / step_ticks))
    pitch = np.full(n_steps, -1, int)
    keep_lower = mono == "lowest"
    for note, start, end in events:
        first = int(round(start / step_ticks))
        last = max(first + 1, int(round(end / step_ticks)))
        for step in range(first, min(last, n_steps)):
            if pitch[step] < 0 or (note < pitch[step] if keep_lower else note > pitch[step]):
                pitch[step] = note
    return pitch


def octave_fit(pitch: np.ndarray, semitones: int = 0, center: int = 40) -> np.ndarray:
    """Transpose voiced pitches, then octave-shift so their median sits near `center`.

    Key is deliberately not normalized — harmony is carried by the chroma, and the
    12-key augmentation is what teaches the model to read it.
    """
    out = pitch.copy()
    voiced = out >= 0
    if not voiced.any():
        return out
    out[voiced] = out[voiced] + semitones
    out[voiced] += 12 * round((center - int(np.median(out[voiced]))) / 12)
    out[voiced] = np.clip(out[voiced], NOTE_MIDI_LO, NOTE_MIDI_HI)
    return out


def notes_to_tokens(pitch_seg: np.ndarray) -> np.ndarray:
    """Per-step pitch (-1 = rest) -> note tokens. The inverse of `tokens_to_notes`."""
    tokens = np.full(len(pitch_seg), NOTE_REST, np.int64)
    previous = -1
    for step, note in enumerate(pitch_seg):
        if note < 0:
            tokens[step] = NOTE_REST
            previous = -1
        elif note == previous:
            tokens[step] = NOTE_SUSTAIN
        else:
            tokens[step] = NOTE_PITCH0 + (note - NOTE_MIDI_LO)
            previous = note
    return tokens


def tokens_to_notes(tokens):
    """Step tokens -> ``(midi, start_step, duration_steps)``. The inverse of `notes_to_tokens`.

    Only the pitch block opens a note. BOS and EOS sit directly above it, so testing for
    `>= NOTE_PITCH0` alone would decode them as phantom pitches 85 and 86.

    A sustain with nothing to extend — at step 0, or after a rest — is read as a rest.
    `generate_notes` masks that case out, so reaching it means a hand-built sequence.
    """
    notes = []
    current = None
    for step, token in enumerate(tokens):
        if token == NOTE_SUSTAIN and current is not None:
            current[2] += 1
        elif NOTE_PITCH0 <= token < NOTE_VOCAB:
            current = [NOTE_MIDI_LO + int(token) - NOTE_PITCH0, step, 1]
            notes.append(current)
        else:
            current = None
    return notes


def write_midi(notes, path: Path, bpm: float = GRID_REF_BPM):
    midi_file = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(int(round(bpm)))))

    ticks_per_step = 480 // 4
    events = []
    for midi, start, duration in notes:
        events.append((start * ticks_per_step, "note_on", midi))
        events.append(((start + duration) * ticks_per_step, "note_off", midi))
    events.sort(key=lambda event: (event[0], event[1] == "note_on"))

    previous_tick = 0
    for tick, kind, midi in events:
        track.append(
            mido.Message(
                kind,
                note=midi,
                velocity=90 if kind == "note_on" else 0,
                time=tick - previous_tick,
            )
        )
        previous_tick = tick
    midi_file.save(path)
