"""Metrics behind the preference charts: did fine-tuning move toward the edits?

This is the measurement code for `preference-shift.html` ("Did fine-tuning move
toward my edits?"). The charts are not training curves — no SFT or DPO run in this
repo ever logged its history — so every number is measured after the fact by loading
a finished checkpoint and scoring it. They are endpoint measurements: where a
checkpoint landed, not the path it took.

THE QUESTION
------------
A fine-tune is supposed to move the model toward the clips you keep. That is testable
directly: score the clip you kept and the clip you replaced under the same checkpoint,
under the exact conditioning both were generated with, and see which one the model
thinks is more likely.

    log P(chosen)  - the edit you exported from the DAW
    log P(rejected) - the original the model produced, which you edited over
    margin          = log P(chosen) - log P(rejected)

A positive margin means the model now prefers your edit. `preference_accuracy` is the
share of pairs with a positive margin: 0.5 is indifference, below 0.5 means the model
actively prefers what it already wrote.

Both sequences are exactly NOTE_STEPS long, so the length bias that affects
preference metrics on variable-length text cannot arise here — a shorter sequence
cannot win by having fewer terms in the sum.

WHY THE SPLIT MATTERS
---------------------
Always read `chosen` and `rejected` separately, never just the margin. A margin can
grow two ways: by raising the chosen side, or by pushing the rejected side down. The
second is suppression, and here the rejected sample is on-policy — a representative
clip you tweaked, not an outlier — so pushing it down damages the general
distribution. In the shipped checkpoints both happen, and they are not the same
event:

    bass  base -> ft2 (SFT)   chosen -132.9 -> -6.9    rejected -73.7 -> -43.9
    bass  ft2  -> ft3 (DPO)   chosen   -6.9 -> -11.2   rejected -43.9 -> -82.1

SFT lifted the chosen side. DPO barely moved it and widened the gap by dropping the
rejected side instead. `PreferenceReport` keeps the halves apart so that is visible.

TRAINING PAIRS ARE NOT HELD OUT
-------------------------------
`training_data/dpo/` holds the pairs ft2 and ft3 were fit on. Scoring a checkpoint
on those measures absorption, not generalization: a log-probability near zero across
65 tokens is ~90% per token on data the model was trained to reproduce. Any claim
about learned taste needs a run the fine-tune never saw. Pass one to
`preference_report` and compare — that is what the second panel of the chart is.

UNCERTAINTY: CLUSTER BY SECTION, NOT BY ROW
-------------------------------------------
Every 4-bar section is emitted 12 times, transposed through all twelve keys (see
`tokenize_sections`). Those 12 rows are the same musical phrase and their scores move
together, so treating them as 12 independent samples understates the standard error
by roughly sqrt(12) ~ 3.5x. `clustered_se` therefore averages within a section first
and takes the standard error across section means. Ten edited sections is n=10, not
n=120, and the honest interval is the wider one.

Rows are keyed (section, shift), but `preference_pairs` drops any pair that tokenizes
identically, so a section can contribute fewer than 12 rows. `keyed_preference_pairs`
below keeps the section id on every row rather than assuming fixed groups of twelve.

REPRODUCING THE CHART
---------------------
    PYTHONPATH=. python -m utils.metric_utils --role bass

prints the table the chart is drawn from: one row per checkpoint, training pairs and
held-out pairs side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch

from utils.config import CKPT_DIR, NOTE_STEPS, NOTE_VOCAB_SIZE, TRAINING_DATA

# The architecture every shipped checkpoint was trained under. Kept here so a
# measurement never silently loads weights into a differently-shaped model.
NOTE_CONFIG = {
    "vocab_size": NOTE_VOCAB_SIZE, "context_length": NOTE_STEPS + 4, "emb_dim": 128,
    "n_heads": 4, "n_layers": 3, "drop_rate": 0.1, "qkv_bias": False,
}

STAGES = ("base", "ft1", "ft2", "ft3")


def _dpo_helpers():
    """Imported lazily: `utils` sits below `train`, so this keeps the layering one-way."""
    from train.finetune_dpo import run_conditioning, sequence_logprob, tokenize_sections
    return run_conditioning, tokenize_sections, sequence_logprob


class Pair(NamedTuple):
    """One (chosen, rejected) row, tagged with the section it came from."""
    section: int
    shift: int
    chosen: tuple
    rejected: tuple


class PreferenceReport(NamedTuple):
    stage: str
    n_sections: int
    n_pairs: int
    chosen: float              # mean log P of the edit
    chosen_se: float           # clustered by section
    rejected: float            # mean log P of what it replaced
    rejected_se: float         # clustered by section
    margin: float              # mean(chosen - rejected); > 0 means the edit is preferred
    margin_se: float           # clustered by section, NOT across transposition rows
    accuracy: float            # share of pairs with a positive margin
    accuracy_se: float


def checkpoint_path(role: str, stage: str) -> Path:
    """`base` -> <role>_notes_gpt.pt, `ftN` -> <role>_notes_gpt_ftN.pt."""
    stem = f"{role}_notes_gpt"
    return CKPT_DIR / (f"{stem}.pt" if stage == "base" else f"{stem}_{stage}.pt")


def load_note_model(path: Path, device=None):
    """Load a checkpoint into NOTE_CONFIG. Returns (eval-mode model, its metadata).

    The metadata is how a checkpoint's provenance is read: SFT writes `edits`, DPO
    writes `pairs` and `method`, and `finetuned_from` names the parent. That is the
    only record of which script produced a given ftN — ft1 and ft2 both name the base
    as their parent, so the ftN numbering is not a chain.
    """
    device = device or torch.device("cpu")
    from model.note_model import HarmonicNoteGPT
    saved = torch.load(path, map_location=device, weights_only=False)
    model = HarmonicNoteGPT(NOTE_CONFIG).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, {k: v for k, v in saved.items() if k not in ("model", "config")}


def keyed_preference_pairs(run_dir: Path, role: str) -> list[Pair]:
    """(chosen, rejected) for every section of a run you actually changed, section id kept.

    Mirrors `train.finetune_dpo.preference_pairs`, which returns bare tuples; the
    section id is what makes `clustered_se` possible. Conditioning is rebuilt from the
    run's metadata.json, so a pair is always scored against the exact kick grid and
    chroma that produced it.
    """
    run_conditioning, tokenize_sections, _ = _dpo_helpers()
    grid64, chromas, _ = run_conditioning(run_dir)
    chosen = tokenize_sections(run_dir / "stems" / f"{role}_edited.mid", role, grid64, chromas)
    rejected = tokenize_sections(run_dir / "stems" / f"{role}.mid", role, grid64, chromas)
    pairs = []
    for (section, shift), take in chosen.items():
        leave = rejected.get((section, shift))
        if leave is None:
            continue
        if np.array_equal(take[1], leave[1]):          # unchanged after tokenization
            continue
        pairs.append(Pair(section, shift, take, leave))
    return pairs


def collect_pairs(run_dirs: Sequence[Path], role: str) -> list[Pair]:
    """Pairs across several runs, with section ids made unique per run."""
    out, offset = [], 0
    for run_dir in run_dirs:
        found = keyed_preference_pairs(run_dir, role)
        out += [p._replace(section=p.section + offset) for p in found]
        offset += max((p.section for p in found), default=-1) + 1
    return out


def pair_logprobs(model, pairs: Sequence[Pair], device=None):
    """Per-pair (log P(chosen), log P(rejected)), summed over the 65 positions."""
    _, _, sequence_logprob = _dpo_helpers()
    device = device or torch.device("cpu")

    def column(side: int, field: int, dtype):
        return torch.tensor(
            np.array([getattr(p, side)[field] for p in pairs]), dtype=dtype, device=device
        )

    xc = column("chosen", 0, torch.long);  yc = column("chosen", 1, torch.long)
    xr = column("rejected", 0, torch.long); yr = column("rejected", 1, torch.long)
    grid = column("chosen", 2, torch.float)   # conditioning is shared by both sides
    chord = column("chosen", 3, torch.float)
    with torch.no_grad():
        return sequence_logprob(model, xc, yc, grid, chord), \
               sequence_logprob(model, xr, yr, grid, chord)


def clustered_se(values: np.ndarray, sections: np.ndarray) -> float:
    """Standard error across sections, after averaging within each section.

    The 12 transpositions of one phrase are not independent draws. Averaging them
    first and taking the SE over section means is the honest interval; the naive
    row-wise SE is roughly sqrt(12) too small. With a single section this is
    undefined and returns nan rather than a falsely tight zero.
    """
    values = np.asarray(values, dtype=np.float64)
    means = np.array([values[sections == s].mean() for s in np.unique(sections)])
    if len(means) < 2:
        return float("nan")
    return float(means.std(ddof=1) / np.sqrt(len(means)))


def preference_report(model, pairs: Sequence[Pair], stage: str = "", device=None) -> PreferenceReport:
    """Score one checkpoint on one set of pairs. See the module docstring for the method."""
    chosen_lp, rejected_lp = pair_logprobs(model, pairs, device)
    chosen = chosen_lp.cpu().numpy().astype(np.float64)
    rejected = rejected_lp.cpu().numpy().astype(np.float64)
    sections = np.array([p.section for p in pairs])
    margin = chosen - rejected
    wins = (margin > 0).astype(np.float64)
    return PreferenceReport(
        stage=stage,
        n_sections=int(len(np.unique(sections))),
        n_pairs=len(pairs),
        chosen=float(chosen.mean()),
        chosen_se=clustered_se(chosen, sections),
        rejected=float(rejected.mean()),
        rejected_se=clustered_se(rejected, sections),
        margin=float(margin.mean()),
        margin_se=clustered_se(margin, sections),
        accuracy=float(wins.mean()),
        accuracy_se=clustered_se(wins, sections),
    )


def stage_series(role: str, run_dirs: Sequence[Path], stages=STAGES, device=None):
    """One PreferenceReport per checkpoint — the series each chart line is drawn from."""
    pairs = collect_pairs(run_dirs, role)
    if not pairs:
        raise SystemExit(f"no changed {role} pairs in {[str(d) for d in run_dirs]}")
    out = []
    for stage in stages:
        model, _ = load_note_model(checkpoint_path(role, stage), device)
        out.append(preference_report(model, pairs, stage, device))
    return out


def training_runs() -> list[Path]:
    """The preference pairs ft2 and ft3 were fit on. NOT held out — see the docstring."""
    return sorted(d for d in (TRAINING_DATA / "dpo").iterdir() if d.is_dir())


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="bass", choices=("bass", "arp"))
    ap.add_argument("--held-out", type=Path, default=None,
                    help="a run directory the fine-tune never saw, scored alongside")
    args = ap.parse_args(argv)

    def table(title, reports):
        print(f"\n{title}   ({reports[0].n_sections} sections, {reports[0].n_pairs} rows)")
        print(f"  {'stage':6}{'logP(edit)':>12}{'logP(orig)':>12}"
              f"{'margin':>10}{'±SE':>8}{'prefers edit':>14}")
        for r in reports:
            print(f"  {r.stage:6}{r.chosen:>12.1f}{r.rejected:>12.1f}"
                  f"{r.margin:>+10.1f}{r.margin_se:>8.1f}{r.accuracy:>13.0%}")

    table("TRAINING PAIRS (absorption)", stage_series(args.role, training_runs()))
    if args.held_out:
        table("HELD OUT (generalization)", stage_series(args.role, [args.held_out]))
    else:
        print("\n  no --held-out run given; training-pair numbers alone measure absorption,")
        print("  not learned taste. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
