"""Specialist 2-step UKG track generator — the top-level UX.

    python -m inference.make_track -i          # interactive: asks genre, key, bpm, weights
    python -m inference.make_track --genre ukg-dark --bpm 140   # scriptable flags
    python -m inference.make_track --base      # force the un-tuned base models
    python -m inference.make_track --request "make a 2 step ukg beat in A minor, dark, \
moody, night time in london, 8 bars long"

Bare `python -m inference.make_track` in a terminal drops into the interactive prompts;
with any args (or no tty) it runs non-interactively from --request/--genre/--bpm.

Weights default to the LATEST fine-tuned checkpoint (checkpoints/<role>_notes_gpt_ftN.pt,
highest N), falling back to base if none — on-policy generation so new hand-edits target
residual taste, not errors SFT already fixed. Use --base for the un-tuned control set,
which the folder name records via its _ftN tag (absent = base).
Parses key / mood / length, maps mood -> chord progression + sampling temps, then
runs the full cascade:

    drum loop from drum_samples/                           [conditioning input]
      -> kick-pocket grid                                   (rhythm skeleton)
    progression from key+mood                               (harmony skeleton)
      -> bass  HarmonicNoteGPT -> MIDI                      [note model]
      -> arp   HarmonicNoteGPT -> MIDI                      [note model]
    -> output/<MMDD-HHMMSS>_<slug>/ (stems/{bass,arp}.mid, metadata.json)

Length beyond 4 bars = independently generated 4-bar sections over the same
progression (A A' form), concatenated into one continuous MIDI performance
per stem.

Output is symbolic. The drum loop is read for its kick-pocket grid only — it
conditions generation and is never mixed in. Load the MIDI into a DAW to voice it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from utils.audio_features import align_to_grid_tempo, detect_bpm, onset_grid
from model.checkpoints import latest_ckpt, load_note_model
from utils.chords import NAMES, color_progression, progression_to_track
from utils.config import (
    CKPT_DIR, DRUM_DIR, GRID_BARS, GRID_REF_BPM, GRID_STEPS, NOTE_BARS, NOTE_BOS,
    NOTE_EOS, NOTE_MIDI_LO, NOTE_PITCH0, NOTE_REST, NOTE_STEPS, OUTPUT_DIR, SAMPLE_RATE,
)
from utils.device import pick_device
from utils.midi_utils import tokens_to_notes, write_midi
from model.note_model import generate_notes, note_chord_cond, note_grid_cond
from utils.scoring import fit_and_lock


def drum_loops() -> list[Path]:
    """Every conditioning loop available in drum_samples/, by name."""
    return sorted(DRUM_DIR.glob("*.wav"))


def default_drum() -> Path:
    """The conditioning loop: the only wav in drum_samples/, else the first by name.

    Resolved lazily rather than at import so an empty folder fails at run time with a
    useful message instead of breaking every import of this module.
    """
    loops = drum_loops()
    if not loops:
        raise SystemExit(
            f"no drum loop in {DRUM_DIR} — drop a 2-step wav there, or pass --drum"
        )
    return loops[0]


PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
      "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}

# mood -> progression template (semitone offset from root, quality) + sampling temps.
# Degrees: dark minor = i-VI-iv-v (no major dominant, stays shadowed);
# neutral minor = i-VI-III-VII (the classic); major = I-vi-IV-V.
# Filter cutoffs and mix gains lived here while the pipeline rendered audio; the
# symbolic path leaves voicing to the DAW.
MOODS = {
    "dark": {
        "degrees_minor": [(0, "m"), (8, ""), (5, "m"), (7, "m")],
        "degrees_major": [(0, ""), (9, "m"), (5, "m"), (7, "")],
        "bass_temp": 1.2, "arp_temp": 1.3,
    },
    "neutral": {
        "degrees_minor": [(0, "m"), (8, ""), (3, ""), (10, "")],
        "degrees_major": [(0, ""), (9, "m"), (5, ""), (7, "")],
        "bass_temp": 1.3, "arp_temp": 1.35,
    },
}
DARK_WORDS = ("dark", "moody", "night", "london", "deep", "heavy", "brooding", "shadow")

# Genre registry. Today the whole cascade is 2-step UK garage, so the only honest
# entries select a mood preset (progression + sampling temps) over the UKG drum
# library. Add a genre by giving it a mood preset, a default drum loop, and its
# native tempo — the CLI menu and metadata pick it up automatically.
# "drum": None means "whatever default_drum() finds in drum_samples/".
GENRES = {
    "ukg-dark":    {"label": "2-step UK garage — dark / moody",     "mood": "dark",    "drum": None, "bpm": 130},
    "ukg-classic": {"label": "2-step UK garage — classic / bright", "mood": "neutral", "drum": None, "bpm": 130},
}
DEFAULT_GENRE = "ukg-dark"


def run_folder(root: Path, slug: str) -> Path:
    """A fresh folder per generation: ``<root>/<MMDD-HHMMSS>_<slug>/``.

    Seconds, not minutes: generation takes a few seconds, so back-to-back runs would
    otherwise share a stamp and be told apart only by a counter. Timestamp first so a
    listing reads chronologically, and so nothing an earlier run wrote is overwritten.

    The counter below is a safety net for the same-second case, which needs two runs
    launched in parallel to happen at all. It trails the slug so those still sort
    beside the run they collided with.
    """
    stamp = time.strftime("%m%d-%H%M%S")
    out = root / f"{stamp}_{slug}"
    attempt = 2
    while out.exists():
        out = root / f"{stamp}_{slug}-{attempt}"
        attempt += 1
    return out


def parse_key(text: str) -> tuple[str, str]:
    """'A minor' / 'F#m' / 'Bb major' / 'c' -> (root, mode). Mode defaults to minor."""
    m = re.match(r"\s*([a-gA-G][#b]?)\s*(minor|min|major|maj|m)?\s*$", text)
    if not m or m.group(1).upper() not in PC:
        raise ValueError(f"unrecognized key: {text!r} (try e.g. 'A minor', 'F#m', 'Bb major')")
    root = m.group(1)[0].upper() + m.group(1)[1:]                 # 'f#' -> 'F#', 'bb' -> 'Bb'
    mode = "major" if (m.group(2) or "m").lower().startswith("maj") else "minor"
    return root, mode


def parse_request(text: str) -> dict:
    """'... in A minor, dark, moody ..., 8 bars long' -> key/mode/bars/mood."""
    t = text.lower()
    m = re.search(r"\b([a-g][#b]?)\s*(minor|min|major|maj|m\b)", t)
    root, mode = ("A", "minor") if not m else (
        m.group(1).upper().replace("B", "b") if len(m.group(1)) > 1 else m.group(1).upper(),
        "major" if m.group(2).startswith("maj") else "minor",
    )
    b = re.search(r"(\d+)\s*bars?", t)
    bars = int(b.group(1)) if b else 8
    bars = max(NOTE_BARS, (bars // NOTE_BARS) * NOTE_BARS)        # multiple of 4
    mood = "dark" if any(w in t for w in DARK_WORDS) else "neutral"
    return {"root": root, "mode": mode, "bars": bars, "mood": mood}


def build_progression(root: str, mode: str, mood: str) -> str:
    prof = MOODS[mood]
    degrees = prof["degrees_minor"] if mode == "minor" else prof["degrees_major"]
    rp = PC[root.upper()]
    names = [NAMES[(rp + off) % 12] + ("m" if q == "m" else "") for off, q in degrees]
    return "-".join(names)


def gen_section(model, grid_tok, chord_tok, temp, seed, device):
    torch.manual_seed(seed)
    out = generate_notes(model, grid_tok, NOTE_BOS, NOTE_EOS, max_steps=NOTE_STEPS,
                         temperature=temp, top_p=0.98, chord_tok=chord_tok, device=device)
    tk = out[0, 1:].cpu().numpy()
    return np.pad(tk, (0, max(0, NOTE_STEPS - len(tk))), constant_values=NOTE_REST)[:NOTE_STEPS]


def best_section(model, grid_tok, chord_tok, chroma, grid, temp, base_seed, device, n_cand=10):
    """Best-of-N: sample candidates, score = harmony fit + kick lock + density sanity.

    Density prefers a real UKG bassline (6-16 onsets per 4 bars) over one droning
    note or machine-gun spam. Cheap on a 621k model; big quality-floor win."""
    best, best_score, best_fl = None, -1e9, (0.0, 0.0)
    for c in range(n_cand):
        tk = gen_section(model, grid_tok, chord_tok, temp, seed=base_seed + c * 101, device=device)
        fit, lock = fit_and_lock(tk, chroma, grid)
        n_on = int((tk >= NOTE_PITCH0).sum())
        density = 1.0 - min(abs(n_on - 10) / 10.0, 1.0)          # peak at ~10 onsets
        score = 2.0 * fit + 1.0 * lock + 0.7 * density + (-2.0 if n_on < 3 else 0.0)
        if score > best_score:
            best, best_score, best_fl = tk, score, (fit, lock)
    return best, best_fl


def resolve_ckpt(role: str, explicit: Path | None, use_sft: bool) -> Path:
    """--<role>-ckpt wins; else latest SFT (default) or the base checkpoint under --base."""
    if explicit is not None:
        return explicit
    return latest_ckpt(role) if use_sft else CKPT_DIR / f"{role}_notes_gpt.pt"


def _ask(prompt: str, default: str, parse):
    """Prompt until parse(raw) succeeds; blank input takes the default."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip() or default
        try:
            return parse(raw)
        except ValueError as e:
            print(f"  {e}")


def prompt_spec() -> dict:
    """Interactive front door: ask genre, key, bpm (+ bars). Returns a track spec."""
    print("\n── make track ──────────────────────────────────────")
    genres = list(GENRES)
    print("genre:")
    for i, k in enumerate(genres, 1):
        print(f"  {i}. {GENRES[k]['label']}")

    def pick_genre(raw):
        if raw.isdigit() and 1 <= int(raw) <= len(genres):
            return genres[int(raw) - 1]
        if raw in GENRES:
            return raw
        raise ValueError(f"choose 1-{len(genres)}")

    def bpm_ok(raw):
        b = int(raw)
        if not 60 <= b <= 200:
            raise ValueError("bpm out of range (60-200)")
        return b

    def yn(raw):
        if raw.lower() in ("y", "yes"):
            return True
        if raw.lower() in ("n", "no"):
            return False
        raise ValueError("answer y or n")

    genre = _ask("  choose", "1", pick_genre)
    root, mode = _ask("key (e.g. A minor, F#m, Bb major)", "A minor", parse_key)
    bpm = _ask("bpm", str(GENRES[genre]["bpm"]), bpm_ok)
    bars = _ask("bars", "8", lambda r: max(NOTE_BARS, (int(r) // NOTE_BARS) * NOTE_BARS))
    bl, al = latest_ckpt("bass"), latest_ckpt("arp")
    if "_ft" in bl.name or "_ft" in al.name:                 # only ask when SFT weights exist
        print(f"  latest SFT: bass={bl.name}  arp={al.name}")
        use_sft = _ask("use latest SFT weights? (n = base)", "Y", yn)
    else:
        use_sft = False                                      # no ft checkpoints -> base is all there is
    return {"genre": genre, "root": root, "mode": mode, "bpm": bpm, "bars": bars,
            "mood": GENRES[genre]["mood"], "drum": GENRES[genre]["drum"], "use_sft": use_sft}


def main(argv=None) -> Path:
    """Generate a track and return the run folder it was written to.

    `argv` defaults to the command line. Callers that are not the CLI — the notebook,
    the Streamlit app — pass a list instead of swapping sys.argv, and take the returned
    path rather than guessing at the newest folder in output/.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="prompt for genre / key / bpm (default when run with no args in a terminal)")
    ap.add_argument("--request", default="8 bar ukg track in A minor, dark",
                    help="natural-language track request (non-interactive)")
    ap.add_argument("--genre", choices=list(GENRES), default=None, help="genre preset (overrides request mood)")
    ap.add_argument("--bpm", type=int, default=None, help="tempo (default: genre's native bpm)")
    ap.add_argument("--drum-bpm", type=float, default=None,
                    help="tempo of the conditioning loop (default: read from its filename, "
                         "else assume it is already at 130)")
    ap.add_argument("--drum", type=Path, default=None,
                    help="2-step drum loop wav (default: the loop in drum_samples/)")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: a new random seed each run. Pass the seed printed by an "
                         "earlier run (and match its device) to reproduce it exactly")
    ap.add_argument("--candidates", type=int, default=10, help="best-of-N per 4-bar section")
    ap.add_argument("--bass-temp", type=float, default=None,
                    help="override the mood's bass sampling temperature (default 1.2-1.3)")
    ap.add_argument("--arp-temp", type=float, default=None,
                    help="override the mood's arp sampling temperature (default 1.3-1.35)")
    ap.add_argument("--bass-ckpt", type=Path, default=None, help="override bass model (e.g. a _ftN checkpoint)")
    ap.add_argument("--arp-ckpt", type=Path, default=None, help="override arp model (e.g. a _ftN checkpoint)")
    ap.add_argument("--base", action="store_true",
                    help="force the un-tuned base models (default: latest SFT _ftN checkpoint)")
    ap.add_argument("--no-arp", action="store_true", help="bass only")
    ap.add_argument("--chords", default="fixed", choices=("fixed", "color"),
                    help="fixed: the mood template's plain triads. color: recolor each "
                         "chord with an in-key 7th or suspension, freshly per section")
    ap.add_argument("--device", default=None, choices=("cpu", "mps", "cuda", "auto"),
                    help="override the sampling device (default: cpu — ~10x faster here)")
    ap.add_argument("--out-root", type=Path, default=OUTPUT_DIR,
                    help="where generated MIDI folders are written (default: output/)")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = ap.parse_args(argv)

    interactive = args.interactive or (not argv and sys.stdin.isatty())
    if interactive:
        if not sys.stdin.isatty():
            raise SystemExit("--interactive needs a terminal (no tty). Use --request/--genre/--bpm instead.")
        spec = prompt_spec()
    else:
        base = parse_request(args.request)
        g = GENRES[args.genre or DEFAULT_GENRE]
        spec = {"genre": args.genre or DEFAULT_GENRE, "root": base["root"], "mode": base["mode"],
                "bars": base["bars"], "mood": g["mood"] if args.genre else base["mood"],
                "bpm": args.bpm or g["bpm"], "drum": g["drum"], "use_sft": not args.base}

    req = {k: spec[k] for k in ("root", "mode", "bars", "mood")}
    # A fixed default seed made every run of the same request produce identical music —
    # generation is deterministic given (seed, device, weights, conditioning). Vary it by
    # default and print it, so runs differ but any one of them can still be reproduced.
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    bpm = float(args.bpm or spec["bpm"])
    genre = spec["genre"]
    drum = args.drum or spec["drum"] or default_drum()
    request_str = (args.request if not interactive else
                   f"{GENRES[genre]['label']}, {req['root']} {req['mode']}, {int(bpm)} bpm, {req['bars']} bars")
    prof = MOODS[req["mood"]]
    bass_temp = args.bass_temp if args.bass_temp is not None else prof["bass_temp"]
    arp_temp = args.arp_temp if args.arp_temp is not None else prof["arp_temp"]
    prog = build_progression(req["root"], req["mode"], req["mood"])
    n_sections = req["bars"] // NOTE_BARS

    # resolve checkpoints before the slug so the folder name records which weights ran
    use_sft = spec["use_sft"] and not args.base            # --base forces base even in interactive
    bass_ckpt = resolve_ckpt("bass", args.bass_ckpt, use_sft)
    arp_ckpt = resolve_ckpt("arp", args.arp_ckpt, use_sft)

    slug = f"{req['root'].lower().replace('#','s')}_{req['mode']}_{req['mood']}_{req['bars']}bar"
    if round(bpm) != int(GRID_REF_BPM):                       # keep 130-BPM folder names stable; tag off-tempo runs
        slug += f"_{round(bpm)}bpm"
    tags = []
    for ck in (bass_ckpt, arp_ckpt):
        m = re.search(r"_(ft\d+)$", ck.stem)
        tag = m.group(1) if m else None                  # base checkpoints -> canonical (untagged) folder
        if tag and tag not in tags:
            tags.append(tag)
    if tags:
        slug += "_" + "-".join(tags)
    out = run_folder(args.out_root, slug)
    (out / "stems").mkdir(parents=True, exist_ok=True)
    print(f"request: {request_str}")
    print(f"parsed:  genre={genre}  key={req['root']} {req['mode']}  mood={req['mood']}  bars={req['bars']}"
          f"  progression={prog}  bpm={bpm:.0f}", flush=True)
    print(f"weights: bass={bass_ckpt.name}  arp={arp_ckpt.name}"
          f"  ({'SFT' if '_ft' in bass_ckpt.name else 'base'})", flush=True)
    print(f"sampling: seed={seed}  temps bass={bass_temp} arp={arp_temp}  "
          f"best-of-{args.candidates}", flush=True)
    if args.no_arp:
        print("stems:   bass only (arp disabled)", flush=True)

    device = pick_device(args.device)

    # ── rhythm skeleton: drum loop -> kick grid (2-bar loop tiled to 4-bar sections).
    # The loop is read purely to condition generation; its audio is never written out.
    # Analysis stays at SAMPLE_RATE so the grid matches what the models trained on.
    drum24, _ = librosa.load(drum, sr=SAMPLE_RATE, mono=True)
    drum_bpm = args.drum_bpm or detect_bpm(drum) or GRID_REF_BPM
    loop_bars = len(drum24) / SAMPLE_RATE / (4 * 60.0 / drum_bpm)
    source = "given" if args.drum_bpm else ("filename" if detect_bpm(drum) else "assumed")
    print(f"drum:    {Path(drum).name}  ({loop_bars:.1f} bars @ {drum_bpm:.0f}, {source})", flush=True)
    if abs(drum_bpm - GRID_REF_BPM) >= 0.01:
        print(f"         stretched to {GRID_REF_BPM:.0f} so its bars fill the analysis window",
              flush=True)
    drum24 = align_to_grid_tempo(drum24, drum_bpm)
    # onset_grid truncates to GRID_FRAMES, so only the first GRID_BARS ever reach the
    # model. Longer loops are not an error, but say so rather than dropping them quietly.
    if loop_bars > GRID_BARS + 0.05:
        print(f"         conditioning on bars 1-{GRID_BARS} only, tiled across each "
              f"4-bar section ({GRID_BARS / loop_bars:.0%} of the file)", flush=True)
    g = onset_grid(drum24)
    grid32 = (np.array([b.max() if b.size else 0.0
                        for b in np.array_split(g, GRID_STEPS)]) > 0.25).astype(np.float32)
    grid64 = np.tile(grid32, NOTE_STEPS // len(grid32))[:NOTE_STEPS]
    grid_tok = note_grid_cond(grid64)

    # ── harmony skeleton: progression -> per-step chroma.
    # Fixed mode reuses one progression for every section, so an 8-bar run is the same
    # 4 bars of harmony twice. Color mode redraws the voicings per section, giving A and
    # A' different colors over the same roots — variation the seed alone cannot reach,
    # since the chroma is conditioning rather than something the model samples.
    chord_rng = random.Random(seed)
    key_pc = PC[req["root"].upper()]
    progs = [prog if args.chords == "fixed"
             else color_progression(prog, key_pc, req["mode"], chord_rng)
             for _ in range(n_sections)]
    chromas = [progression_to_track(p, NOTE_STEPS) for p in progs]
    chord_toks = [note_chord_cond(c) for c in chromas]
    if args.chords != "fixed":
        for i, p in enumerate(progs, 1):
            print(f"  section {i}: {p}", flush=True)

    # ── note stems: N independent 4-bar sections per stem (A A' form)
    metrics = {}
    roles = [("bass", bass_ckpt, bass_temp)]
    if not args.no_arp:
        roles.append(("arp", arp_ckpt, arp_temp))
    for role, ckpt, temp in roles:
        model = load_note_model(ckpt, device)
        role_off = 0 if role == "bass" else 31          # fixed (hash() is salted per-process)
        notes, fits, locks = [], [], []
        for s in range(n_sections):
            tk, (f, l) = best_section(model, grid_tok, chord_toks[s], chromas[s], grid64, temp,
                                      base_seed=seed * 1000 + s * 7 + role_off,
                                      device=device, n_cand=args.candidates)
            fits.append(f); locks.append(l)
            sec_notes = tokens_to_notes(tk)
            notes += [(m, st + s * NOTE_STEPS, d) for m, st, d in sec_notes]
        metrics[role] = {"chord_fit": round(float(np.mean(fits)), 3),
                         "kick_lock": round(float(np.mean(locks)), 3),
                         "notes": len(notes)}
        write_midi(notes, out / "stems" / f"{role}.mid", bpm=bpm)
        print(f"  {role}: {len(notes)} notes over {n_sections} sections  "
              f"fit={metrics[role]['chord_fit']}  lock={metrics[role]['kick_lock']}", flush=True)

    total_sec = req["bars"] * 4 * 60.0 / bpm

    meta = {
        "request": request_str,
        "parsed": req,
        "genre": genre,
        "bpm": bpm,
        "progression": prog,
        "progressions": progs,          # per section; equal to prog in fixed mode
        "chords": args.chords,
        "sections": n_sections,
        "duration_sec": round(total_sec, 2),
        "drum_loop": str(Path(drum).name),          # conditioning source, not mixed audio
        "drum_bpm": drum_bpm,
        # device is part of the sampling record: a seed reproduces on the same device
        # only — MPS and CPU diverge within a few steps from identical state.
        "sampling": {"bass_temp": bass_temp, "arp_temp": arp_temp, "top_p": 0.98,
                     "seed": seed, "candidates_per_section": args.candidates,
                     "device": str(device)},
        "validation": metrics,
        "models": {r: c.name for r, c, *_ in roles},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    written = "  ".join(f"{r}.mid" for r, *_ in roles)
    print(f"\nmidi -> {out / 'stems'}  ({written})")
    print(f"  {total_sec:.1f}s, {req['bars']} bars @ {bpm:.0f}   metadata.json written")
    return out


if __name__ == "__main__":
    main()
