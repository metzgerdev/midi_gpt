"""Generate a track.

    uv run --frozen streamlit run ui/app.py

Output is MIDI. Download the stems and voice them in a DAW — nothing here plays audio,
and the piano roll lives in your sequencer, which is better at it.
"""
from __future__ import annotations

import json

import streamlit as st

from inference.make_track import GENRES, MOODS
from ui.state import capture, checkpoints, drum_loops
from utils.config import CKPT_DIR

st.title("Generate")

loops = drum_loops()
if not loops:
    st.error("No drum loop in `drum_samples/`. Drop a 2-step wav there to generate.")
    st.stop()

with st.sidebar:
    st.subheader("Track")
    root = st.selectbox("Key", ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
    mode = st.radio("Mode", ["minor", "major"], horizontal=True)
    bars = st.select_slider("Bars", [4, 8, 12, 16], value=8,
                            help="Generated as independent 4-bar sections over one progression")
    genre = st.selectbox("Genre", list(GENRES), format_func=lambda g: GENRES[g]["label"])
    bpm = st.number_input("BPM", 100, 180, GENRES[genre]["bpm"])

    st.subheader("Conditioning")
    drum = st.selectbox("Drum loop", loops, format_func=lambda p: p.name,
                        help="Read for its kick grid only. The audio is never mixed in.")
    chords = st.radio("Chords", ["fixed", "color"], horizontal=True,
                      help="color: recolor each chord with an in-key 7th or suspension")

    st.subheader("Sampling")
    mood = GENRES[genre]["mood"]
    bass_temp = st.slider("Bass temperature", 0.5, 2.0, MOODS[mood]["bass_temp"], 0.05)
    arp_temp = st.slider("Arp temperature", 0.5, 2.0, MOODS[mood]["arp_temp"], 0.05)
    candidates = st.slider("Candidates per section", 1, 30, 10,
                           help="Best-of-N on chord fit, kick lock and note density")
    fixed_seed = st.checkbox("Fixed seed")
    seed = st.number_input("Seed", 0, 10**9, 7) if fixed_seed else None

    st.subheader("Weights")
    tuned = [p.stem.rsplit("_", 1)[-1] for p in checkpoints("bass") if "_ft" in p.stem]
    weights = st.selectbox("Checkpoint", ["latest", "base"] + tuned)
    no_arp = st.checkbox("Bass only")

argv = [
    "--request", f"{bars} bar track in {root} {mode}",
    "--genre", genre, "--bpm", str(int(bpm)), "--drum", str(drum),
    "--chords", chords, "--candidates", str(candidates),
    "--bass-temp", str(bass_temp), "--arp-temp", str(arp_temp),
]
if seed is not None:
    argv += ["--seed", str(int(seed))]
if no_arp:
    argv += ["--no-arp"]
if weights == "base":
    argv += ["--base"]
elif weights != "latest":
    for role in ("bass", "arp"):
        argv += [f"--{role}-ckpt", str(CKPT_DIR / f"{role}_notes_gpt_{weights}.pt")]

if st.button("Generate", type="primary", width='stretch'):
    from inference.make_track import main

    with st.spinner("Generating…"):
        st.session_state["run"], st.session_state["log"] = capture(main, argv)

run = st.session_state.get("run")
if run is None:
    st.info("Set the track up on the left, then generate. A run takes a couple of seconds.")
    if st.session_state.get("log"):
        st.error(st.session_state["log"])
    st.stop()

meta = json.loads((run / "metadata.json").read_text())
st.success(f"`{run.name}`")

left, right = st.columns([2, 3])
with left:
    st.caption("Track")
    st.write(f"**{meta['progression']}** · {meta['bpm']:.0f} BPM · "
             f"{meta['parsed']['bars']} bars in {meta['sections']} section(s)")
    if meta.get("progressions") and len(set(meta["progressions"])) > 1:
        st.caption("per section: " + "  |  ".join(meta["progressions"]))
    st.write(f"seed `{meta['sampling']['seed']}` · kick from `{meta['drum_loop']}`")

with right:
    st.caption("Validation")
    st.dataframe(
        [{"role": role,
          "model": meta["models"][role],
          "notes": scores["notes"],
          "chord fit": scores["chord_fit"],
          "kick lock": scores["kick_lock"]}
         for role, scores in meta["validation"].items()],
        hide_index=True, width='stretch',
    )

st.caption("Download the stems and drag them into your DAW.")
for role in meta["validation"]:
    stem = run / "stems" / f"{role}.mid"
    st.download_button(f"{role}.mid", stem.read_bytes(), file_name=f"{run.name}_{role}.mid",
                       mime="audio/midi")

st.caption(f"Run folder: `{run}`")
with st.expander("Log"):
    st.code(st.session_state.get("log", ""))
