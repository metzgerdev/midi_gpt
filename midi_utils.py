"""Conversion between note tokens, MIDI events, and MIDI files."""
from __future__ import annotations

from pathlib import Path

import mido
import numpy as np

from config import GRID_REF_BPM, NOTE_MIDI_LO, NOTE_PITCH0, NOTE_SUSTAIN


def tokens_to_notes(tokens, key_pc: int):
    """Convert step tokens to ``(midi, start_step, duration_steps)`` notes."""
    notes = []
    current = None
    for step, token in enumerate(tokens):
        if token == NOTE_SUSTAIN and current is not None:
            current[2] += 1
        elif token >= NOTE_PITCH0:
            midi = NOTE_MIDI_LO + (token - NOTE_PITCH0) + key_pc
            current = [int(np.clip(midi, 0, 127)), step, 1]
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
