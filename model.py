"""Checkpoint-compatible note model and autoregressive sampler."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import CHORD_DIM, CLAP_DIM, NOTE_REST, NOTE_SUSTAIN
from gpt_model import GPTModel


def note_grid_cond(grid) -> np.ndarray:
    """Convert a per-step kick grid to the BOS-prefixed model alignment."""
    grid = np.asarray(grid, dtype=np.float32)
    return np.concatenate([grid, grid[-1:]]).astype(np.float32)


def note_chord_cond(chroma) -> np.ndarray:
    """Convert per-step chroma to the BOS-prefixed model alignment."""
    chroma = np.asarray(chroma, dtype=np.float32)
    return np.concatenate([chroma, chroma[-1:]], axis=0).astype(np.float32)


class HarmonicNoteGPT(GPTModel):
    """GPT conditioned per step on rhythm and chord chroma.

    ``cond_proj`` is retained even though make_track uses ``cond=None`` because
    it is present in the trained checkpoint state dictionaries.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cond_proj = nn.Linear(CLAP_DIM, cfg["emb_dim"])
        self.grid_proj = nn.Linear(1, cfg["emb_dim"])
        self.chord_proj = nn.Linear(CHORD_DIM, cfg["emb_dim"])

    def forward(self, in_idx, cond=None, cond_seq=None, chord_seq=None):
        _, seq_len = in_idx.shape
        x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        if cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(1)
        if cond_seq is not None:
            x = x + self.grid_proj(cond_seq.unsqueeze(-1))
        if chord_seq is not None:
            x = x + self.chord_proj(chord_seq)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def forbidden_next(previous_token: int, bos_id: int) -> list[int]:
    """Token ids that cannot legally follow `previous_token`.

    BOS opens the sequence and never recurs. SUSTAIN extends the note before it, so it is
    illegal at the start and after a rest: `tokens_to_notes` has to discard such a token,
    which silently shortens the clip the model thought it was writing.
    """
    forbidden = [bos_id]
    if previous_token in (bos_id, NOTE_REST):
        forbidden.append(NOTE_SUSTAIN)
    return forbidden


@torch.no_grad()
def generate_notes(
    model,
    grid_tok,
    bos_id,
    eos_id,
    max_steps,
    temperature=1.0,
    top_k=None,
    top_p=None,
    chord_tok=None,
    device=None,
):
    """Sample one pitch/rest/sustain token per sixteenth-note step."""
    model.eval()
    device = device or next(model.parameters()).device
    grid_t = torch.as_tensor(grid_tok, dtype=torch.float32, device=device)
    chord_t = None if chord_tok is None else torch.as_tensor(
        chord_tok, dtype=torch.float32, device=device
    )
    idx = torch.tensor([[bos_id]], device=device)

    for _ in range(max_steps):
        pos = idx.shape[1]
        kwargs = {} if chord_t is None else {
            "chord_seq": chord_t[:pos].view(1, pos, CHORD_DIM)
        }
        logits = model(
            idx,
            cond=None,
            cond_seq=grid_t[:pos].view(1, pos),
            **kwargs,
        )[:, -1, :]
        logits[:, forbidden_next(int(idx[0, -1]), bos_id)] = float("-inf")

        if temperature <= 0.0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            if top_p:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probabilities = torch.softmax(sorted_logits, dim=-1)
                drop = torch.cumsum(probabilities, dim=-1) - probabilities > top_p
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_indices, sorted_logits
                )
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)

        if (nxt == eos_id).all():
            break
        idx = torch.cat([idx, nxt], dim=1)

    return idx
