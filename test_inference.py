import importlib
import random
import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest
import torch

from chords import (
    infer_chord, infer_chord_track,
    NAMES, PC, chord_chroma, color_progression, parse_progression,
    progression_to_track, scale_pcs,
)
from config import (
    ROLE_CENTER, ROLE_MONO,
    NOTE_BOS,
    NOTE_EOS,
    NOTE_MIDI_LO,
    NOTE_PITCH0,
    NOTE_REST,
    NOTE_STEPS,
    NOTE_SUSTAIN,
    NOTE_VOCAB_SIZE,
)
from make_track import build_progression, parse_key, parse_request
from midi_utils import (
    midi_to_step_grid, notes_to_tokens, octave_fit, tokens_to_notes, write_midi,
)
from model import HarmonicNoteGPT, note_chord_cond, note_grid_cond


def test_note_vocabulary_is_contiguous():
    assert NOTE_BOS == NOTE_VOCAB_SIZE - 2
    assert NOTE_EOS == NOTE_VOCAB_SIZE - 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A minor", ("A", "minor")),
        ("F#m", ("F#", "minor")),
        ("Bb major", ("Bb", "major")),
        ("c", ("C", "minor")),
    ],
)
def test_parse_key(text, expected):
    assert parse_key(text) == expected


def test_request_and_progression():
    request = parse_request("12 bar garage track in F# minor, dark and moody")
    assert request == {
        "root": "F#",
        "mode": "minor",
        "bars": 12,
        "mood": "dark",
    }
    assert build_progression("A", "minor", "dark") == "Am-F-Dm-Em"


def test_progression_and_model_conditioning_shapes():
    chroma = progression_to_track("Am-F-C-G", NOTE_STEPS)
    grid = np.zeros(NOTE_STEPS, np.float32)
    assert chroma.shape == (NOTE_STEPS, 12)
    assert note_chord_cond(chroma).shape == (NOTE_STEPS + 1, 12)
    assert note_grid_cond(grid).shape == (NOTE_STEPS + 1,)
    assert np.all(chroma.sum(axis=1) == 3)


def test_tokens_to_notes_handles_sustain_and_rest():
    tokens = [NOTE_PITCH0, NOTE_SUSTAIN, NOTE_REST, NOTE_PITCH0 + 12]
    assert tokens_to_notes(tokens, key_pc=0) == [
        [NOTE_MIDI_LO, 0, 2],
        [NOTE_MIDI_LO + 12, 3, 1],
    ]


def test_write_midi_round_trips_notes(tmp_path):
    """The MIDI file is now the deliverable, so its contents are worth asserting."""
    path = tmp_path / "bass.mid"
    write_midi([(48, 0, 4), (55, 8, 2)], path, bpm=130)
    assert path.exists() and path.stat().st_size > 0

    midi = mido.MidiFile(path)
    played = [m.note for track in midi.tracks for m in track if m.type == "note_on"]
    assert played == [48, 55]


@pytest.mark.parametrize("role", ["bass", "arp"])
def test_latest_checkpoint_loads_strictly(role):
    checkpoint = Path(f"checkpoints/{role}_notes_gpt_ft3.pt")
    if not checkpoint.exists():
        pytest.skip(f"local inference asset is absent: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HarmonicNoteGPT(saved["config"])
    model.load_state_dict(saved["model"], strict=True)


@pytest.mark.parametrize("mode", ["minor", "major"])
def test_colored_chords_never_add_out_of_key_tones(mode):
    """Coloring may only add tones the key already contains.

    The dark-major template deliberately borrows a minor iv, so the base progression
    itself can sit outside the key; the invariant is that coloring never makes that
    worse, not that every chord is diatonic.
    """
    for root in NAMES:
        for mood in ("dark", "neutral"):
            base = build_progression(root, mode, mood)
            scale = scale_pcs(PC[root.upper()], mode)
            base_chords = parse_progression(base)
            for trial in range(20):
                colored = parse_progression(
                    color_progression(base, PC[root.upper()], mode, random.Random(trial))
                )
                assert len(colored) == len(base_chords)
                for (b_root, b_qual), (c_root, c_qual) in zip(base_chords, colored):
                    assert c_root == b_root, "coloring must not move a chord's root"
                    base_out = set(np.flatnonzero(chord_chroma(b_root, b_qual)).tolist()) - scale
                    col_out = set(np.flatnonzero(chord_chroma(c_root, c_qual)).tolist()) - scale
                    assert col_out <= base_out, f"{base} -> {colored} left the key"


def test_color_progression_is_seeded():
    same = [color_progression("Am-F-Dm-Em", 9, "minor", random.Random(3)) for _ in range(3)]
    assert len(set(same)) == 1, "a fixed rng seed must give a fixed progression"
    varied = {color_progression("Am-F-Dm-Em", 9, "minor", random.Random(s)) for s in range(30)}
    assert len(varied) > 5, "different seeds should reach different colorings"


def test_midi_token_round_trip_is_lossless():
    """tokens -> MIDI -> tokens must be exact, or DPO would pair a clip against itself
    and invent preference signal from a tokenizer artifact."""
    tokens = np.array([11, 1, 1, 0, 19, 1, 1, 1, 0, 0, 14, 1, 1, 0, 0, 0])
    path = Path(tempfile.mkdtemp()) / "round.mid"
    write_midi(tokens_to_notes(tokens, key_pc=0), path, bpm=130)

    pitch = midi_to_step_grid(mido.MidiFile(path), mono="lowest")
    recovered = notes_to_tokens(pitch)
    # trailing rests are not written to MIDI, so the file is short; the rest is exact
    assert np.array_equal(recovered, tokens[:len(recovered)])

    padded = np.concatenate([pitch, np.full(len(tokens) - len(pitch), -1, int)])
    assert np.array_equal(notes_to_tokens(padded), tokens), "padding must restore trailing rests"


def test_mono_reduction_picks_the_right_voice():
    """bass keeps the fundamental, arp keeps the top line — as the corpus was built."""
    path = Path(tempfile.mkdtemp()) / "chord.mid"
    write_midi([(40, 0, 4), (47, 0, 4), (52, 0, 4)], path, bpm=130)
    assert midi_to_step_grid(mido.MidiFile(path), mono="lowest")[0] == 40
    assert midi_to_step_grid(mido.MidiFile(path), mono="highest")[0] == 52


@pytest.mark.parametrize("role,mono,center", [("bass", "lowest", 40), ("arp", "highest", 60)])
def test_role_reduction_matches_the_corpus(role, mono, center):
    """If these drift, an edited clip tokenizes differently from the original it pairs with."""
    assert ROLE_MONO[role] == mono
    assert ROLE_CENTER[role] == center


def test_octave_fit_centres_without_changing_intervals():
    pitch = np.array([60, 64, 67, -1, 60])
    fitted = octave_fit(pitch, semitones=0, center=40)
    voiced = pitch >= 0
    assert np.all(np.diff(fitted[voiced]) == np.diff(pitch[voiced])), "intervals preserved"
    assert fitted[3] == -1, "rests untouched"
    assert abs(int(np.median(fitted[voiced])) - 40) <= 6


@pytest.mark.parametrize("module", ["mine_corpus", "train_notes", "finetune_sft", "finetune_dpo"])
def test_training_scripts_import(module):
    """The retraining path must not rot when the inference code is refactored."""
    importlib.import_module(module)


def test_inferred_chord_matches_the_notes_it_was_built_from():
    """Self-supervised harmony: a C minor line must label itself C minor."""
    line = np.full(16, -1)
    line[[0, 4, 8, 12]] = [48, 51, 55, 48]              # C3 Eb3 G3 C3
    chroma = infer_chord_track(line, 1, 16)[0]
    assert set(np.flatnonzero(chroma).tolist()) == {0, 3, 7}, "C, Eb, G"


def test_infer_chord_is_deterministic_when_templates_tie():
    """One pitch class ties many templates; the winner must at least be stable."""
    histogram = np.zeros(12, np.float32); histogram[9] = 8
    first = infer_chord(histogram)
    assert np.array_equal(first, infer_chord(histogram.copy()))
    assert first[9] == 1.0, "the sounding pitch class is always in the chosen chord"
