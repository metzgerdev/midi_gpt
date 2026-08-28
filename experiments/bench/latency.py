"""Measure generation latency on the shipped checkpoints. CPU vs MPS, no training.

The README and `utils/device.py` quote 632 ms per candidate on MPS against 60 ms on CPU,
and ~25 s vs ~2.4 s for an 8-bar track. Those numbers had no benchmark behind them — there
was not a single `perf_counter` call in the repo — and the counterintuitive half of the
claim is exactly the half someone will ask you to defend. This produces them.

    PYTHONPATH=. python experiments/bench/latency.py            # -> latency.json
    PYTHONPATH=. python experiments/bench/latency.py --reps 5   # quicker, noisier

Three levels, because they answer different questions:

  candidate   one 4-bar section = 64 sequential single-token forward passes. The unit
              the device comparison is about.
  track       best-of-10 x 2 sections x 2 roles = 40 candidates, the sampling cost of
              an 8-bar run.
  end-to-end  what a user waits for: drum-loop load, BPM detection, tempo alignment,
              onset grid, both models loaded from disk, generation, MIDI written.

Reports median and p95 rather than the mean — the first run through a fresh model is
slower than the rest and a mean hides that behind a number that describes no single run.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from model.checkpoints import latest_ckpt, load_note_model
from model.note_model import note_chord_cond, note_grid_cond
from utils.chords import progression_to_track
from utils.config import BASE, NOTE_STEPS

BENCH_PROGRESSION = "Am-F-C-G"
HERE = Path(__file__).resolve().parent


def available_devices(requested: list[str] | None) -> list[str]:
    """The devices worth timing on this machine."""
    if requested:
        return requested
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def conditioning():
    """A fixed grid and chroma, so every timed run does identical work.

    Deliberately not read from a drum loop: file IO and BPM detection belong to the
    end-to-end number, not to the per-candidate one.
    """
    grid = np.zeros(NOTE_STEPS, dtype=np.float32)
    grid[::4] = 1.0                                   # four-on-the-floor, a plain case
    chroma = progression_to_track(BENCH_PROGRESSION, NOTE_STEPS)
    return note_grid_cond(grid), note_chord_cond(chroma), grid, chroma


def sync(device: str) -> None:
    """Make the timer see work that an accelerator would otherwise still be doing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def time_candidates(model, grid_tok, chord_tok, device, reps, warmup):
    """Seconds per single 4-bar candidate."""
    from inference.make_track import gen_section

    for i in range(warmup):
        gen_section(model, grid_tok, chord_tok, 0.9, seed=i, device=torch.device(device))
    sync(device)

    samples = []
    for i in range(reps):
        start = time.perf_counter()
        gen_section(model, grid_tok, chord_tok, 0.9, seed=1000 + i,
                    device=torch.device(device))
        sync(device)
        samples.append(time.perf_counter() - start)
    return samples


def time_track(models, cond, device, reps, candidates, sections):
    """Seconds for the sampling half of an 8-bar, two-stem run."""
    from inference.make_track import best_section

    grid_tok, chord_tok, grid, chroma = cond
    samples = []
    for i in range(reps):
        start = time.perf_counter()
        for model in models:
            for s in range(sections):
                best_section(model, grid_tok, chord_tok, chroma, grid, 0.9,
                             base_seed=i * 1000 + s * 7, device=torch.device(device),
                             n_cand=candidates)
        sync(device)
        samples.append(time.perf_counter() - start)
    return samples


def time_end_to_end(device, reps, out_root):
    """Seconds from request string to MIDI on disk, everything included."""
    from inference.make_track import main as make_track

    samples = []
    for i in range(reps):
        start = time.perf_counter()
        make_track([
            "--request", "make a 2 step ukg beat in A minor, dark, 8 bars long",
            "--device", device, "--seed", str(i), "--out-root", str(out_root),
        ])
        samples.append(time.perf_counter() - start)
    return samples


def summarise(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "median_ms": round(statistics.median(ordered) * 1000, 1),
        "p95_ms": round(ordered[max(0, round(0.95 * len(ordered)) - 1)] * 1000, 1),
        "min_ms": round(ordered[0] * 1000, 1),
        "max_ms": round(ordered[-1] * 1000, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="*", default=None,
                    help="default: cpu, plus mps/cuda when available")
    ap.add_argument("--reps", type=int, default=20, help="timed candidate runs")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--track-reps", type=int, default=3)
    ap.add_argument("--e2e-reps", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=10, help="best-of-N per section")
    ap.add_argument("--sections", type=int, default=2, help="2 sections = 8 bars")
    ap.add_argument("--skip-end-to-end", action="store_true",
                    help="skip the run that writes MIDI to disk")
    ap.add_argument("--out", type=Path, default=HERE / "latency.json")
    args = ap.parse_args(argv)

    cond = conditioning()
    grid_tok, chord_tok, _, _ = cond
    scratch = BASE / "output" / "_bench"

    results = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__,
        },
        "settings": {"candidates": args.candidates, "sections": args.sections,
                     "reps": args.reps, "warmup": args.warmup},
        "checkpoints": {r: latest_ckpt(r).name for r in ("bass", "arp")},
        "devices": {},
    }

    for device in available_devices(args.devices):
        print(f"=== {device} ===", flush=True)
        models = [load_note_model(latest_ckpt(r), torch.device(device))
                  for r in ("bass", "arp")]

        candidate = summarise(time_candidates(models[0], grid_tok, chord_tok, device,
                                              args.reps, args.warmup))
        print(f"  candidate    median {candidate['median_ms']:>9.1f} ms  "
              f"p95 {candidate['p95_ms']:.1f} ms", flush=True)

        track = summarise(time_track(models, cond, device, args.track_reps,
                                     args.candidates, args.sections))
        print(f"  track        median {track['median_ms']:>9.1f} ms  "
              f"p95 {track['p95_ms']:.1f} ms", flush=True)

        entry = {"candidate": candidate, "track_sampling": track}

        if not args.skip_end_to_end:
            e2e = summarise(time_end_to_end(device, args.e2e_reps, scratch))
            print(f"  end-to-end   median {e2e['median_ms']:>9.1f} ms  "
                  f"p95 {e2e['p95_ms']:.1f} ms", flush=True)
            entry["end_to_end"] = e2e

        results["devices"][device] = entry

    # The headline claim is a ratio, so compute it rather than leaving it to be eyeballed.
    if "cpu" in results["devices"] and len(results["devices"]) > 1:
        cpu = results["devices"]["cpu"]["candidate"]["median_ms"]
        results["cpu_speedup"] = {
            d: round(v["candidate"]["median_ms"] / cpu, 2)
            for d, v in results["devices"].items() if d != "cpu"
        }
        print(f"\nCPU is faster per candidate by: {results['cpu_speedup']}")

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
