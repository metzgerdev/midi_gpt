"""midi_gpt — generate MIDI, then teach the model your taste.

    uv run --frozen streamlit run ui/app.py

Output is MIDI. Download the stems and voice them in a DAW; nothing here plays audio,
and the piano roll lives in your sequencer, which is better at it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# The views import utils.*, model.*, inference.* and train.*, which live one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="midi_gpt", page_icon="🎹", layout="wide")

st.navigation([
    st.Page("views/generate.py", title="Generate", icon="🎹", default=True),
    st.Page("views/finetune.py", title="Fine-tune", icon="🎛"),
]).run()
