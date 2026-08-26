import importlib
import json
import random
import re
import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest
import torch

from utils.chords import (
    infer_chord, infer_chord_track,
    NAMES, PC, chord_chroma, color_progression, parse_progression,
    progression_to_track, scale_pcs,
)
from utils.config import (
    CKPT_DIR,
    ROLE_CENTER, ROLE_MONO,
    NOTE_BOS,
    NOTE_EOS,
    NOTE_MIDI_HI,
    NOTE_MIDI_LO,
    NOTE_PITCH0,
    NOTE_REST,
    NOTE_STEPS,
    NOTE_SUSTAIN,
    NOTE_VOCAB_SIZE,
    TRAINING_DATA,
)
from inference.make_track import build_progression, parse_key, parse_request
from model.checkpoints import load_checkpoint
from utils.midi_utils import (
    midi_to_step_grid, note_onsets, notes_to_tokens, octave_fit, tokens_to_notes,
    write_midi,
)
from model.note_model import (
    HarmonicNoteGPT, forbidden_next, generate_notes, note_chord_cond, note_grid_cond,
)


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
    assert tokens_to_notes(tokens) == [
        [NOTE_MIDI_LO, 0, 2],
        [NOTE_MIDI_LO + 12, 3, 1],
    ]


def test_tokens_to_notes_inverts_notes_to_tokens_over_the_whole_range():
    """The decoder is unconditionally the inverse of the encoder.

    It used to take a `key_pc` transpose that `notes_to_tokens` had no counterpart for,
    so any caller passing a nonzero key silently shifted the clip on the way out.
    """
    pitches = np.arange(NOTE_MIDI_LO, NOTE_MIDI_HI + 1)
    assert len(pitches) == 61, "the pitch block is C1..C6 inclusive"
    decoded = tokens_to_notes(notes_to_tokens(pitches))
    assert decoded == [[int(pitch), step, 1] for step, pitch in enumerate(pitches)]


def test_tokens_to_notes_ignores_sequence_markers():
    """BOS and EOS sit just above the pitch block, and used to decode as pitches 85 and 86."""
    assert tokens_to_notes([NOTE_BOS, NOTE_EOS]) == []
    assert tokens_to_notes([NOTE_BOS, NOTE_PITCH0, NOTE_EOS]) == [[NOTE_MIDI_LO, 1, 1]]


def test_orphan_sustain_never_opens_a_note():
    """A sustain with no note to extend is a rest, not a phantom onset."""
    assert tokens_to_notes([NOTE_SUSTAIN, NOTE_SUSTAIN]) == []
    assert tokens_to_notes([NOTE_PITCH0, NOTE_REST, NOTE_SUSTAIN]) == [[NOTE_MIDI_LO, 0, 1]]


STUB_CONFIG = {"vocab_size": NOTE_VOCAB_SIZE, "context_length": NOTE_STEPS + 4,
               "emb_dim": 32, "n_heads": 2, "n_layers": 1, "drop_rate": 0.0,
               "qkv_bias": False}


class ScriptedNoteModel(HarmonicNoteGPT):
    """Wants `script[step]` at every step, so greedy sampling is fully deterministic.

    Every other id sits at a uniform floor, so when the mask forbids the scripted choice
    the lowest legal id wins — NOTE_REST. That makes an over-broad mask show up as rests
    where a real token was expected, instead of passing as if it were correct.
    """

    def __init__(self, script):
        super().__init__(STUB_CONFIG)
        self.script = script

    def forward(self, in_idx, **_):
        batch, seq_len = in_idx.shape
        logits = torch.full((batch, seq_len, NOTE_VOCAB_SIZE), -10.0)
        logits[:, -1, self.script[min(seq_len - 1, len(self.script) - 1)]] = 10.0
        return logits


def sample_scripted(script, max_steps=6):
    """The tokens a scripted model actually emits, BOS stripped."""
    generated = generate_notes(
        ScriptedNoteModel(script),
        note_grid_cond(np.zeros(NOTE_STEPS, np.float32)),
        NOTE_BOS,
        NOTE_EOS,
        max_steps=max_steps,
        temperature=0.0,
        chord_tok=note_chord_cond(np.zeros((NOTE_STEPS, 12), np.float32)),
    )
    return generated[0, 1:].tolist()


@pytest.mark.parametrize("previous", [NOTE_BOS, NOTE_REST])
def test_sustain_is_forbidden_with_nothing_to_extend(previous):
    assert NOTE_SUSTAIN in forbidden_next(previous, NOTE_BOS)


@pytest.mark.parametrize("previous", [NOTE_PITCH0, NOTE_SUSTAIN])
def test_sustain_is_allowed_after_a_sounding_note(previous):
    assert NOTE_SUSTAIN not in forbidden_next(previous, NOTE_BOS)


@pytest.mark.parametrize("previous", [NOTE_BOS, NOTE_REST, NOTE_SUSTAIN, NOTE_PITCH0])
def test_bos_is_never_legal_again(previous):
    assert NOTE_BOS in forbidden_next(previous, NOTE_BOS)


@pytest.mark.parametrize("script", [[NOTE_SUSTAIN], [NOTE_REST, NOTE_SUSTAIN]])
def test_sampling_refuses_an_orphan_sustain(script):
    """A sustain the decoder would have to discard must never be sampled at all.

    Discarding one silently shortens the clip: the model commits its remaining steps to
    extending a note that was never written.
    """
    assert NOTE_SUSTAIN not in sample_scripted(script)


def test_sampling_still_allows_a_legitimate_sustain():
    """The mask must block only the illegal case — a held note is how duration is spelled."""
    assert sample_scripted([NOTE_PITCH0, NOTE_SUSTAIN]) == [NOTE_PITCH0] + [NOTE_SUSTAIN] * 5


def test_sampling_never_re_emits_bos():
    assert NOTE_BOS not in sample_scripted([NOTE_BOS])


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
    saved = load_checkpoint(checkpoint, torch.device("cpu"))
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
    write_midi(tokens_to_notes(tokens), path, bpm=130)

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


@pytest.mark.parametrize("module", ["train.mine_corpus", "train.train_notes",
                                    "train.finetune_sft", "train.finetune_dpo"])
def test_training_scripts_import(module):
    """The retraining path must not rot when the inference code is refactored."""
    importlib.import_module(module)


def test_every_shipped_preference_run_can_rebuild_its_conditioning():
    """The tracked example pairs are the only worked example of the fine-tuning loop.

    Their loop is archived under training_data/drum_loops/ while DRUM_DIR points at
    drum_samples/, so looking in one place alone left every one of them unusable in a
    fresh clone — `run_conditioning` raised before a single pair was built.
    """
    from train.finetune_dpo import find_drum_loop

    runs = sorted((TRAINING_DATA / "dpo").glob("*/metadata.json"))
    if not runs:
        pytest.skip("no tracked preference runs")
    for metadata_path in runs:
        name = json.loads(metadata_path.read_text())["drum_loop"]
        assert find_drum_loop(name) is not None, (
            f"{metadata_path.parent.name} needs {name}, which is in neither loop directory"
        )


def test_missing_drum_loop_is_still_reported():
    """Searching two directories must not turn a missing loop into a silent pass."""
    from train.finetune_dpo import find_drum_loop

    assert find_drum_loop("no_such_loop_9f3a.wav") is None


def test_make_track_takes_argv_and_returns_the_run_folder(tmp_path):
    """Callers that are not the CLI need both halves of this.

    Without them the notebook had to swap sys.argv and then guess at the newest folder
    in output/, which is a race as soon as two runs overlap.
    """
    from inference.make_track import main

    if not (CKPT_DIR / "bass_notes_gpt.pt").exists():
        pytest.skip("local inference asset is absent")

    run = main(["--request", "4 bar UKG track in A minor, dark", "--seed", "7",
                "--candidates", "1", "--no-arp", "--out-root", str(tmp_path)])
    assert run.parent == tmp_path, "wrote outside --out-root"
    assert (run / "stems" / "bass.mid").exists()
    assert json.loads((run / "metadata.json").read_text())["parsed"]["bars"] == 4


def test_note_onsets_records_restrikes_that_the_pitch_array_loses():
    """The grid's whole job: a re-struck note is a SUSTAIN in the tokens, so the only
    place the attack survives is the onset track."""
    path = Path(tempfile.mkdtemp()) / "restrike.mid"
    write_midi([(40, 0, 2), (40, 2, 2), (40, 4, 2), (47, 6, 2)], path, bpm=130)
    midi = mido.MidiFile(path)
    pitch = midi_to_step_grid(midi, mono="lowest")

    onsets = note_onsets(midi, len(pitch))
    changes = np.array([p >= 0 and p != (pitch[i - 1] if i else -1)
                        for i, p in enumerate(pitch)], np.float32)

    assert int(onsets.sum()) == 4, "every strike is an onset"
    assert int(changes.sum()) == 2, "the pitch array only sees two changes"
    assert int((notes_to_tokens(pitch) >= NOTE_PITCH0).sum()) == 2, "tokens agree with pitch"


@pytest.mark.slow          # scans for the source MIDI outside the repo
def test_mine_corpus_grid_rule_reproduces_the_shipped_corpus():
    """The .npz in training_data/ are the record of what the checkpoints learned from.

    Deriving the grid from the mono-reduced pitch array instead of the note events made
    it identical to the token onsets, which is both a different signal and a strictly
    less informative one. Pinned here against the data itself.
    """
    import hashlib

    from utils.config import ROLE_MONO

    corpus = sorted((TRAINING_DATA / "bass_notes_midi").glob("*_k0.npz"))
    if not corpus:
        pytest.skip("mined corpus is absent (gitignored); nothing to pin against")

    sources = {}
    for candidate in Path.home().joinpath("Documents/Code").rglob("*bass*.mid*"):
        try:
            digest = hashlib.md5(candidate.read_bytes()).hexdigest()[:10]
        except OSError:
            continue
        sources.setdefault(digest, candidate)

    pattern = re.compile(r"^(?P<stem>.+)_(?P<digest>[0-9a-f]{10})__(?P<chunk>\d+)_k0$")
    checked = 0
    for npz_path in corpus:
        matched = pattern.match(npz_path.stem)
        source = sources.get(matched["digest"]) if matched else None
        if source is None:
            continue
        midi = mido.MidiFile(source)
        pitch = midi_to_step_grid(midi, mono=ROLE_MONO["bass"])
        if pitch is None:
            continue
        lo = int(matched["chunk"]) * NOTE_STEPS
        onsets = note_onsets(midi, len(pitch))
        if lo + NOTE_STEPS > len(onsets):
            continue
        assert np.array_equal(onsets[lo:lo + NOTE_STEPS], np.load(npz_path)["grid"]), (
            f"{npz_path.name} does not match the rule that produced it"
        )
        checked += 1
        if checked >= 40:
            break
    if not checked:
        pytest.skip("no corpus example could be matched to its source MIDI")


def test_training_from_scratch_refuses_to_replace_an_existing_checkpoint(tmp_path):
    """Training writes to the checkpoint every other script loads by default.

    It used to save on every validation improvement, so an interrupted run left a
    half-trained model where the finished one belonged — and said nothing.
    """
    from train.train_notes import main

    existing = tmp_path / "bass_notes_gpt.pt"
    existing.write_bytes(b"not really a checkpoint")
    with pytest.raises(SystemExit) as excinfo:
        main(["--role", "bass", "--out", str(existing), "--data-dir", str(tmp_path)])
    assert "already exists" in str(excinfo.value)
    assert existing.read_bytes() == b"not really a checkpoint", "the file was touched"


def test_training_checks_the_checkpoint_before_loading_the_corpus(tmp_path):
    """The refusal must come first, or it costs an epoch to find out."""
    from train.train_notes import main

    existing = tmp_path / "taken.pt"
    existing.write_bytes(b"x")
    missing_corpus = tmp_path / "no_such_corpus"
    with pytest.raises(SystemExit) as excinfo:
        main(["--role", "bass", "--out", str(existing), "--data-dir", str(missing_corpus)])
    assert "already exists" in str(excinfo.value), (
        "the corpus error won the race, so the guard runs after the data loads"
    )


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
