"""Render the KL-vs-drift figure from kl_history.json. The generator, not a screenshot.

`.gitignore` warns that `figures/` holds "screenshots with no generator left in the
repo", which makes every published figure unreproducible. This is one of the generators
that fixes that, and it plots data that was already sitting at the repo root unused:
`kl_history.json` is five iterated-DPO arms, and it is the evidence for the claim that
the adaptive KL penalty holds drift to a budget.

    PYTHONPATH=. python experiments/figures/kl_drift.py     # -> figures/kl-drift.png

Reads only. No checkpoint is loaded and nothing is trained.

The arms are the experiment:

  baseline          no KL penalty, reference re-anchored each round — drift compounds
  fixed-ref only    reference pinned to round 0, still no penalty
  kl-fixed L=0.5/2/5  a fixed KL weight, three strengths

What the figure actually shows, which is not quite what I expected to plot:

  - The penalty works. Unpenalised, KL from the base policy reaches 0.205 nats by round
    8 and is still climbing. Any fixed weight flattens it — L=2 and L=5 both land at
    0.017, an order of magnitude lower, and the curve is flat rather than slowing.
  - The quality metrics do NOT show the damage. Over nine rounds `chord_fit` and
    `kick_lock` wander inside a narrow band with no consistent ordering between arms,
    and the unpenalised arm ends *highest* on `kick_lock`. So this experiment
    demonstrates that the penalty controls drift; it does not demonstrate that
    uncontrolled drift costs anything measurable at this horizon.

The second point is why the right-hand panels are labelled as unresolved rather than as
evidence. Nine rounds on ten preference pairs is a short horizon, and these two metrics
measure constraint satisfaction rather than whether the output is any good.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                  # no display in CI or over ssh
import matplotlib.pyplot as plt

from utils.config import BASE

HISTORY = BASE / "kl_history.json"
OUT = BASE / "figures" / "kl-drift.png"

# Colour-blind-safe, and ordered so the unpenalised arms read as the warning cases.
COLOURS = {
    "baseline": "#d55e00",
    "fixed-ref only": "#e69f00",
    "kl-fixed L=0.5": "#56b4e9",
    "kl-fixed L=2": "#0072b2",
    "kl-fixed L=5": "#009e73",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", type=Path, default=HISTORY)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args(argv)

    if not args.history.exists():
        raise SystemExit(f"no such history file: {args.history}")
    history = json.loads(args.history.read_text())

    panels = [
        ("kl", "KL from the base policy (nats)",
         "The penalty works: unpenalised drift keeps climbing, penalised arms flatten."),
        ("chord_fit", "chord_fit",
         "No consistent cost visible over 9 rounds."),
        ("kick_lock", "kick_lock",
         "Same — the unpenalised arm even ends highest here."),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(14, 4.2), constrained_layout=True)

    for axis, (key, ylabel, subtitle) in zip(axes, panels):
        for arm, rounds in history.items():
            if not rounds or key not in rounds[0]:
                continue
            axis.plot([r["round"] for r in rounds], [r[key] for r in rounds],
                      marker="o", markersize=3, linewidth=1.6,
                      color=COLOURS.get(arm), label=arm)
        axis.set_xlabel("DPO round")
        axis.set_ylabel(ylabel)
        axis.set_title(subtitle, fontsize=9, loc="left", color="#555")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle(
        "Iterated DPO: the KL penalty controls drift. Whether drift costs quality "
        "is not resolved at this horizon.",
        fontsize=12, x=0.005, ha="left",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {args.out}")

    for arm, rounds in history.items():
        if rounds and "kl" in rounds[0]:
            print(f"  {arm:<18} final KL {rounds[-1]['kl']:8.3f}  "
                  f"over {len(rounds)} rounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
