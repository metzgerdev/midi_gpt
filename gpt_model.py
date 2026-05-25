"""Minimal GPT backbone required by the trained note-model checkpoints.

Adapted from ``llms_from_scratch.ch03`` and ``llms_from_scratch.ch04`` in
Sebastian Raschka's *Build a Large Language Model From Scratch* repository.
Copyright (c) Sebastian Raschka, licensed under Apache License 2.0.

"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
    ):
        super().__init__()
        if d_out % num_heads:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        batch_size, n_tokens, _ = x.shape
        keys = self.W_key(x).view(
            batch_size, n_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        queries = self.W_query(x).view(
            batch_size, n_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = self.W_value(x).view(
            batch_size, n_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask = self.mask.bool()[:n_tokens, :n_tokens]
        attention_scores.masked_fill_(mask, -torch.inf)
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5,
            dim=-1,
        )
        attention_weights = self.dropout(attention_weights)
        context = (attention_weights @ values).transpose(1, 2)
        context = context.reshape(batch_size, n_tokens, self.d_out)
        return self.out_proj(context)


class LayerNorm(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized + self.shift


class GELU(nn.Module):
    def forward(self, x):
        coefficient = torch.sqrt(x.new_tensor(2.0 / torch.pi))
        return 0.5 * x * (
            1.0
            + torch.tanh(
                coefficient * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dimension = config["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            GELU(),
            nn.Linear(4 * embedding_dimension, embedding_dimension),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"],
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_resid = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.drop_resid(self.att(self.norm1(x))) + shortcut
        shortcut = x
        return self.drop_resid(self.ff(self.norm2(x))) + shortcut


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(
            config["context_length"],
            config["emb_dim"],
        )
        self.drop_emb = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False,
        )

    def forward(self, token_indices):
        _, sequence_length = token_indices.shape
        positions = torch.arange(sequence_length, device=token_indices.device)
        x = self.tok_emb(token_indices) + self.pos_emb(positions)
        x = self.trf_blocks(self.drop_emb(x))
        return self.out_head(self.final_norm(x))
