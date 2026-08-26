"""The retired CLAP path is gone and the shipped checkpoints still work.

`HarmonicNoteGPT.cond_proj` was a `Linear(512, 128)` for a clip embedding nothing in this
repo could produce. Every call passed `cond=None`, so the layer never entered the graph and
never took a gradient — but all eight checkpoints were written while it existed and still
carry its 65,664 weights. Removing the layer therefore breaks every strict load unless the
keys are dropped on the way in.

Two claims are worth holding onto, and this file covers both:

1. Every shipped checkpoint still loads, strictly, into the smaller model.
2. Removing the layer changed no output. Because `cond_proj` never contributed to the
   forward pass, greedy generation must be *identical* to what the pre-removal code
   produced. `fixtures/clap_removal_golden.json` was captured on the commit before the
   removal and is the reference — if a future change to the model perturbs generation,
   test_generation_matches_pre_removal_golden fails.

The golden fixture is greedy (temperature=0), so it has no dependence on seeding or on
torch's RNG implementation, and it is generated on CPU to keep it device-independent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.checkpoints import RETIRED_KEYS, load_checkpoint, load_note_model
from model.note_model import (
    HarmonicNoteGPT, generate_notes, note_chord_cond, note_grid_cond,
)
from utils.chords import progression_to_track
from utils.config import CKPT_DIR, NOTE_BOS, NOTE_EOS, NOTE_STEPS

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "clap_removal_golden.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text())

# Linear(512, 128): 512*128 weights + 128 bias.
COND_PROJ_PARAMS = 512 * 128 + 128
CPU = torch.device("cpu")


def checkpoints() -> list[Path]:
    return sorted(CKPT_DIR.glob("*_notes_gpt*.pt"))


def conditioning():
    """The exact conditioning the golden fixture was captured under."""
    grid = np.zeros(NOTE_STEPS, np.float32)
    grid[::4] = 1.0                                     # four_on_the_floor
    chroma = progression_to_track(GOLDEN["progression"], NOTE_STEPS)
    return note_grid_cond(grid), note_chord_cond(chroma)


def greedy(model, device=CPU) -> list[int]:
    grid_tok, chord_tok = conditioning()
    out = generate_notes(model, grid_tok, NOTE_BOS, NOTE_EOS, max_steps=NOTE_STEPS,
                         temperature=0.0, chord_tok=chord_tok, device=device)
    return out[0, 1:].cpu().numpy().tolist()


# --- the layer is actually gone ---------------------------------------------------

def test_model_has_no_cond_proj():
    model = HarmonicNoteGPT(_any_config())
    assert not hasattr(model, "cond_proj")
    assert not any(k.startswith("cond_proj") for k in model.state_dict())


def test_forward_no_longer_accepts_cond():
    """The parameter is removed, not merely ignored — a stale `cond=` must fail loudly."""
    model = HarmonicNoteGPT(_any_config())
    idx = torch.tensor([[NOTE_BOS]])
    with pytest.raises(TypeError):
        model(idx, cond=torch.zeros(1, 512))


def test_clap_removal_dropped_exactly_65664_parameters():
    model = HarmonicNoteGPT(_any_config())
    now = sum(p.numel() for p in model.parameters())
    before = GOLDEN["checkpoints"]["bass_notes_gpt_ft3.pt"]["n_params_with_cond_proj"]
    assert before - now == COND_PROJ_PARAMS == 65_664
    assert now == 621_184


# --- the shipped checkpoints still load ------------------------------------------

@pytest.mark.parametrize("ckpt", checkpoints(), ids=lambda p: p.name)
def test_shipped_checkpoint_still_carries_the_retired_keys(ckpt):
    """Guards the test below from passing vacuously.

    If the checkpoints are ever re-saved without the retired keys, the stripping in
    `load_checkpoint` stops being exercised and this test says so rather than letting
    the coverage quietly lapse.
    """
    raw = torch.load(ckpt, map_location=CPU, weights_only=False)
    assert set(RETIRED_KEYS) <= set(raw["model"]), (
        f"{ckpt.name} no longer carries {RETIRED_KEYS}; load_checkpoint's strip is untested"
    )


@pytest.mark.parametrize("ckpt", checkpoints(), ids=lambda p: p.name)
def test_load_checkpoint_strips_the_retired_keys(ckpt):
    saved = load_checkpoint(ckpt, CPU)
    assert not any(k.startswith("cond_proj") for k in saved["model"])


@pytest.mark.parametrize("ckpt", checkpoints(), ids=lambda p: p.name)
def test_every_shipped_checkpoint_loads_strictly(ckpt):
    """The regression the removal risked: RuntimeError on unexpected key(s)."""
    model = load_note_model(ckpt, CPU)
    assert isinstance(model, HarmonicNoteGPT)
    assert not model.training


@pytest.mark.parametrize("ckpt", checkpoints(), ids=lambda p: p.name)
def test_model_keeps_the_config_it_was_built_under(ckpt):
    """finetune_sft and finetune_dpo re-save with `model.cfg`.

    They used to hold the raw checkpoint dict open to reach `saved["config"]`; now that
    the load goes through one helper, the architecture has to survive on the model. If
    this drifts, a fine-tune writes a checkpoint stamped with the wrong shape.
    """
    saved = load_checkpoint(ckpt, CPU)
    assert load_note_model(ckpt, CPU).cfg == saved["config"]


def test_stripping_is_idempotent_for_checkpoints_written_after_the_removal(tmp_path):
    """A checkpoint saved by the new code has no retired keys; loading must still work."""
    model = load_note_model(checkpoints()[0], CPU)
    fresh = tmp_path / "fresh_notes_gpt.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg}, fresh)

    saved = load_checkpoint(fresh, CPU)
    assert not any(k.startswith("cond_proj") for k in saved["model"])
    assert greedy(load_note_model(fresh, CPU)) == greedy(model)


def test_a_genuinely_unexpected_key_still_raises(tmp_path):
    """The strip is narrow. Only the retired keys are forgiven; strict=True still bites."""
    model = load_note_model(checkpoints()[0], CPU)
    state = model.state_dict()
    state["not_a_real_layer.weight"] = torch.zeros(3)
    corrupt = tmp_path / "corrupt_notes_gpt.pt"
    torch.save({"model": state, "config": model.cfg}, corrupt)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_note_model(corrupt, CPU)


# --- the model still produces what it used to ------------------------------------

@pytest.mark.parametrize("ckpt", checkpoints(), ids=lambda p: p.name)
def test_generation_matches_pre_removal_golden(ckpt):
    """The whole point: dropping a layer that never ran changes nothing it produced."""
    expected = GOLDEN["checkpoints"][ckpt.name]["greedy_tokens"]
    assert greedy(load_note_model(ckpt, CPU)) == expected


@pytest.mark.parametrize("ckpt", checkpoints(), ids=lambda p: p.name)
def test_first_step_logits_match_pre_removal_golden(ckpt):
    """Tighter than the token check: argmax hides small numeric drift, this does not."""
    model = load_note_model(ckpt, CPU)
    grid_tok, chord_tok = conditioning()
    with torch.no_grad():
        logits = model(
            torch.tensor([[NOTE_BOS]]),
            cond_seq=torch.tensor(grid_tok[:1]).view(1, 1),
            chord_seq=torch.tensor(chord_tok[:1]).view(1, 1, 12),
        )[0, -1].numpy()
    digest = hashlib.sha256(np.round(logits, 4).tobytes()).hexdigest()
    assert digest == GOLDEN["checkpoints"][ckpt.name]["first_step_logits_sha256"]


def test_generation_is_still_conditioned_on_the_chord():
    """A guard against the removal having severed a live path by accident.

    If chord conditioning had been cut along with the dead one, generation would be
    identical across two unrelated progressions. It must not be.
    """
    model = load_note_model(CKPT_DIR / "bass_notes_gpt_ft3.pt", CPU)
    grid_tok = note_grid_cond(np.tile([1, 0, 0, 0], NOTE_STEPS // 4).astype(np.float32))

    def under(progression):
        chord = note_chord_cond(progression_to_track(progression, NOTE_STEPS))
        out = generate_notes(model, grid_tok, NOTE_BOS, NOTE_EOS, max_steps=NOTE_STEPS,
                             temperature=0.0, chord_tok=chord, device=CPU)
        return out[0, 1:].cpu().numpy().tolist()

    assert under("Am-F-C-G") != under("Cm-G#-Fm-Gm")


def _any_config() -> dict:
    """The architecture the shipped checkpoints were trained under."""
    return load_checkpoint(checkpoints()[0], CPU)["config"]
