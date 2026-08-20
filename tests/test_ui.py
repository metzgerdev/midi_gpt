"""The Streamlit UI: its helpers, and that each view renders without raising.

The render tests exist because a crash in a view is invisible from the browser. Streamlit
sends widget events over a websocket, so a failed script shows no HTTP error and no
network activity — the page simply stops updating, which reads as a dead button. The
traceback only reaches the terminal that launched the server.
"""
from __future__ import annotations

import json

import mido
import pytest
import torch

from ui.state import Run, capture, chain_rows, checkpoints, differs_after_tokenizing, runs
from utils.config import BASE, CKPT_DIR
from utils.midi_utils import write_midi


# ---------------------------------------------------------------- helpers


def test_capture_returns_the_result_and_what_was_printed():
    def noisy():
        print("hello")
        return 42

    result, log = capture(noisy)
    assert result == 42
    assert "hello" in log


def test_capture_survives_a_script_that_exits():
    """make_track and finetune_dpo raise SystemExit on bad input. The page must not die."""
    def bails():
        print("partway through")
        raise SystemExit("no drum loop")

    result, log = capture(bails)
    assert result is None
    assert "partway through" in log and "no drum loop" in log


def test_checkpoints_are_base_first_then_ftn_in_order():
    found = [p.name for p in checkpoints("bass")]
    if not found:
        pytest.skip("no local checkpoints")
    assert found[0] == "bass_notes_gpt.pt"
    numbers = [int(n.rsplit("_ft", 1)[1].split(".")[0]) for n in found[1:]]
    assert numbers == sorted(numbers), f"ftN out of order: {found}"


def test_runs_skips_unreadable_metadata(tmp_path):
    good = tmp_path / "0101-000000_good"
    (good / "stems").mkdir(parents=True)
    (good / "metadata.json").write_text(json.dumps({"created": "2026-01-01 00:00:00"}))
    bad = tmp_path / "0101-000001_bad"
    bad.mkdir()
    (bad / "metadata.json").write_text("{ not json")

    found = runs(tmp_path)
    assert [r.name for r in found] == ["0101-000000_good"]


def test_runs_are_newest_first(tmp_path):
    for name, created in [("a", "2026-01-01 00:00:00"), ("b", "2026-06-01 00:00:00")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text(json.dumps({"created": created}))
    assert [r.name for r in runs(tmp_path)] == ["b", "a"]


# ---------------------------------------------------------------- the chain table


def test_chain_rows_are_all_text(tmp_path, monkeypatch):
    """The regression behind a dead-looking Run DPO button.

    st.dataframe serializes through Arrow, which cannot type an object column holding
    both numbers and a placeholder. Checkpoints predating the recipe fields have no
    kl_final or pairs, so mixing one of those with a newer one crashed the render —
    after training had already completed and written a checkpoint.
    """
    import ui.state as state

    monkeypatch.setattr(state, "CKPT_DIR", tmp_path)
    config = {"vocab_size": 65, "context_length": 68, "emb_dim": 8,
              "n_heads": 2, "n_layers": 1, "drop_rate": 0.0, "qkv_bias": False}
    torch.save({"model": {}, "config": config}, tmp_path / "bass_notes_gpt.pt")
    torch.save({"model": {}, "config": config, "edits": 4},          # old: no kl_final
               tmp_path / "bass_notes_gpt_ft1.pt")
    torch.save({"model": {}, "config": config, "method": "dpo",      # new: has both
                "kl_final": 0.3439, "pairs": 10, "anchor": "bass_notes_gpt.pt"},
               tmp_path / "bass_notes_gpt_ft2.pt")

    rows = chain_rows("bass")
    assert len(rows) == 3
    for row in rows:
        for column, value in row.items():
            assert isinstance(value, str), f"{column} is {type(value).__name__}, not str"
    assert rows[1]["KL"] == "—" and rows[2]["KL"] == "0.3439"


def test_chain_rows_survive_arrow_serialization(tmp_path, monkeypatch):
    """Assert the invariant where it actually bit, not just on the row types."""
    pa = pytest.importorskip("pyarrow")
    import ui.state as state

    monkeypatch.setattr(state, "CKPT_DIR", tmp_path)
    config = {"vocab_size": 65, "context_length": 68, "emb_dim": 8,
              "n_heads": 2, "n_layers": 1, "drop_rate": 0.0, "qkv_bias": False}
    torch.save({"model": {}, "config": config}, tmp_path / "arp_notes_gpt.pt")
    torch.save({"model": {}, "config": config, "kl_final": 0.5, "pairs": 3},
               tmp_path / "arp_notes_gpt_ft1.pt")

    rows = chain_rows("arp")
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    pa.table({k: pa.array(v) for k, v in columns.items()})      # raised before the fix


# ---------------------------------------------------------------- edit detection


@pytest.mark.parametrize(
    ("change", "expected"),
    [("none", False), ("velocity", False), ("pitch", True), ("step", True)],
)
def test_differs_after_tokenizing_matches_what_dpo_will_pair(tmp_path, change, expected):
    """Velocity and sub-16th timing vanish in the tokenizer, so DPO cannot learn from
    them. The UI warns about that at upload time, and must agree with the pairing."""
    original = tmp_path / "bass.mid"
    write_midi([(40, 0, 4), (43, 8, 4)], original, bpm=130)

    edited = tmp_path / "bass_edited.mid"
    if change == "none":
        edited.write_bytes(original.read_bytes())
    elif change == "velocity":
        midi = mido.MidiFile(original)
        for track in midi.tracks:
            for message in track:
                if message.type == "note_on" and message.velocity > 0:
                    message.velocity = 30
        midi.save(edited)
    elif change == "pitch":
        write_midi([(45, 0, 4), (43, 8, 4)], edited, bpm=130)
    else:
        write_midi([(40, 2, 4), (43, 8, 4)], edited, bpm=130)

    assert differs_after_tokenizing(original, edited, "bass") is expected


def test_run_reports_which_roles_are_edited(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "stems").mkdir(parents=True)
    (run_dir / "metadata.json").write_text("{}")
    write_midi([(40, 0, 4)], run_dir / "stems" / "bass.mid", bpm=130)
    write_midi([(40, 0, 4)], run_dir / "stems" / "arp.mid", bpm=130)
    write_midi([(45, 0, 4)], run_dir / "stems" / "bass_edited.mid", bpm=130)

    run = Run(run_dir, {})
    assert run.roles() == ["bass", "arp"]
    assert run.edited_roles() == ["bass"]


# ---------------------------------------------------------------- the views


@pytest.mark.parametrize("view", ["ui/views/generate.py", "ui/views/finetune.py"])
def test_view_renders_without_raising(view):
    """A view that raises looks identical to one that is merely slow."""
    from streamlit.testing.v1 import AppTest

    if not (CKPT_DIR / "bass_notes_gpt.pt").exists():
        pytest.skip("local inference asset is absent")

    app = AppTest.from_file(str(BASE / view), default_timeout=90).run()
    assert not app.exception, f"{view} raised: {app.exception}"


def test_generate_view_offers_the_controls_a_run_needs():
    from streamlit.testing.v1 import AppTest

    if not (CKPT_DIR / "bass_notes_gpt.pt").exists():
        pytest.skip("local inference asset is absent")

    app = AppTest.from_file(str(BASE / "ui/views/generate.py"),
                            default_timeout=90).run()
    labels = {w.label for w in list(app.selectbox) + list(app.slider) + list(app.radio)}
    assert {"Key", "Mode", "Genre", "Drum loop", "Checkpoint"} <= labels
    assert any(b.label == "Generate" for b in app.button)


def test_page_finishes_rendering_after_an_upload(tmp_path):
    """The bug behind a Run DPO button that did nothing, silently.

    An uploader keeps its value across reruns, so the branch that saves the file runs
    again on the rerun a button click triggers. Calling st.rerun() there restarts the
    script before the button handler is reached, and the click is lost — no output, no
    error, no network activity, because widget events travel over a websocket.

    Asserting the page renders to the bottom after an upload catches it without running
    a real training round: an aborted script never reaches the chain tables.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(BASE / "ui/views/finetune.py"), default_timeout=120).run()
    if not app.file_uploader or not any(b.label == "Run DPO" for b in app.button):
        pytest.skip("needs a generated run on disk")

    midi = tmp_path / "bass_edited.mid"
    write_midi([(45, 0, 4), (43, 8, 4)], midi, bpm=130)
    app.file_uploader[0].upload("bass_edited.mid", midi.read_bytes(), "audio/midi")
    app.run()

    assert not app.exception, f"raised after upload: {app.exception}"
    assert any(b.label == "Run DPO" for b in app.button), "the button never rendered"
    assert len(app.dataframe) == 2, (
        "the chain tables below the uploader are missing, so the script aborted "
        "partway — a click would be swallowed the same way"
    )
