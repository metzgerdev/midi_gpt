"""Teach the model your taste.

Generate a track, edit the MIDI in a DAW, bring the edit back here. DPO raises the
likelihood of what you played and lowers it for what the model produced, bounded by a
KL budget against a fixed anchor so repeated rounds cannot run away.
"""
from __future__ import annotations

import streamlit as st

from ui.state import (
    capture, chain_rows, checkpoints, differs_after_tokenizing, runs,
)

st.title("Fine-tune on your edits")

found = runs()
if not found:
    st.info("No runs yet. Generate a track first.")
    st.stop()

st.subheader("1. Bring an edit back")
st.caption(
    "Open a run's `bass.mid` in your DAW, change what you would have played differently, "
    "and export it back. Change a note's **pitch** or **step** — velocity and sub-16th "
    "timing cannot be represented in the token vocabulary, so those edits are invisible."
)

labels = {r.name: f"{r.name}   ·   {r.meta.get('progression', '?')}"
          f"{'   ✎ ' + ', '.join(r.edited_roles()) if r.edited_roles() else ''}"
          for r in found}
chosen = st.selectbox("Run", [r.name for r in found], format_func=labels.get)
run = next(r for r in found if r.name == chosen)

col_a, col_b = st.columns(2)
with col_a:
    role = st.radio("Role", run.roles(), horizontal=True)
    original = run.stem(role)
    st.download_button(f"Download {role}.mid to edit", original.read_bytes(),
                       file_name=f"{run.name}_{role}.mid", mime="audio/midi")
with col_b:
    upload = st.file_uploader(f"Upload the edited {role}.mid", type=["mid", "midi"])
    if upload is not None:
        run.edit(role).write_bytes(upload.getvalue())
        st.success(f"Saved as `stems/{role}_edited.mid`")
        st.rerun()

edited = run.edited_roles()
st.write(f"Edits present in this run: **{', '.join(edited) if edited else 'none'}**")

# Say so here rather than after a training run that does nothing. The comparison is the
# same tokenizer DPO uses, so it agrees with what the pairing will decide.
for present in edited:
    if not differs_after_tokenizing(run.stem(present), run.edit(present), present):
        st.warning(
            f"`{present}_edited.mid` tokenizes identically to `{present}.mid`, so DPO "
            f"will find no pair in it. Change a note's **pitch** or **step position** — "
            f"velocity and sub-16th timing cannot be represented."
        )

st.divider()
st.subheader("2. Train")

with st.sidebar:
    st.subheader("DPO settings")
    train_role = st.radio("Role", ["bass", "arp", "both"], horizontal=True)
    kl_target = st.slider("KL budget per run", 0.0, 0.20, 0.05, 0.01,
                          help="How far this run may move the model. 0 disables the KL term.")
    epochs = st.slider("Epochs", 1, 30, 15)
    beta = st.slider("Beta", 0.05, 1.0, 0.1, 0.05,
                     help="Scales the preference reward. It does not bound drift — the KL "
                          "term does.")
    anchor = st.selectbox("Anchor", ["base (recommended)", "current checkpoint"],
                          help="What KL is measured against. The base keeps drift measured "
                               "from one place; re-anchoring lets it compound.")

st.caption(f"Trains on every run under `output/` that has a `_edited.mid`, "
           f"not just the one selected above.")

if st.button("Run DPO", type="primary", width='stretch'):
    from train.finetune_dpo import main

    argv = ["--role", train_role, "--kl-target", str(kl_target),
            "--epochs", str(epochs), "--beta", str(beta)]
    if anchor.startswith("current"):
        argv += ["--anchor-ckpt", str(checkpoints(train_role if train_role != "both"
                                                  else "bass")[-1])]
    with st.spinner("Training…"):
        _, log = capture(main, argv)
    st.session_state["dpo_log"] = log

if st.session_state.get("dpo_log"):
    log = st.session_state["dpo_log"]
    if "nothing to learn from" in log:
        st.warning(
            "**No pairs to train on.** DPO learns from the difference between what the "
            "model produced and what you changed it into. An unedited file gives a zero "
            "margin and no gradient, so there is nothing to move toward.\n\n"
            "Change a note's **pitch** or **step position**. Velocity and sub-16th timing "
            "are not representable in the token vocabulary, so those edits tokenize "
            "identically to the original."
        )
    # Show every line. Filtering to a keyword allowlist hid exactly the messages that
    # explain a run doing nothing, which is when the user most needs them.
    for line in log.splitlines():
        if not line.strip():
            continue
        if "WARNING" in line:
            st.warning(line)
        else:
            st.write(f"`{line}`")

st.divider()
st.subheader("3. The chain")
st.caption("Each fine-tune writes a new checkpoint; make_track uses the highest by default.")

for role in ("bass", "arp"):
    st.write(f"**{role}**")
    st.dataframe(chain_rows(role), hide_index=True, width='stretch')
