from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q, k shaped (B, nhead, L, head_dim). cos, sin: (L, head_dim//2)."""
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0, max_positions: int = 4096):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.theta = float(theta)
        self.max_positions = int(max_positions)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin_for_len(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_positions:
            raise ValueError(f"seq_len={seq_len} exceeds rope.max_positions={self.max_positions}")
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin


class RopeSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE; exposes ``out_proj`` for LoRA."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        rope_theta: float,
        max_positions: int,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got {d_model} / {nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta, max_positions=max_positions)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, D); key_padding_mask: (B, L) True = padding (ignore), same as nn.TransformerEncoder
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.nhead, self.head_dim).transpose(1, 2)
        cos, sin = self.rope.cos_sin_for_len(seqlen, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if key_padding_mask is not None:
            valid = ~key_padding_mask
            attn_mask = valid.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None
        drop = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=drop,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.out_proj(attn_out)


class RopeTransformerEncoderLayer(nn.Module):
    """Post-norm encoder layer (``norm_first=False``) matching ``nn.TransformerEncoderLayer`` default."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        rope_theta: float,
        max_positions: int,
    ):
        super().__init__()
        self.self_attn = RopeSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            rope_theta=rope_theta,
            max_positions=max_positions,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = src
        sa = self.self_attn(x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout1(sa))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class RopeTransformerEncoder(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask)
        return out
