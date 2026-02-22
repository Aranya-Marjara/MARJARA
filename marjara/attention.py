"""
Grouped Query Attention (GQA) with RoPE, QK-norm, sliding window, and KV cache support.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HParams
from .rope import RotaryPos
from .cache import CacheThingy


class SimpleNorm(nn.Module):
    """RMSNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class Attn(nn.Module):
    def __init__(self, cfg: HParams, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        self.sliding_window = cfg.sliding_window

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryPos(self.head_dim, max_seq_len=cfg.context_length * 2, theta=cfg.rope_theta)

        # QK-norm (improves stability at scale)
        self.q_norm = SimpleNorm(self.head_dim)
        self.k_norm = SimpleNorm(self.head_dim)

        self.cache: Optional[CacheThingy] = None

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        batch, kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(batch, kv_heads, n_rep, seq_len, head_dim)
            .reshape(batch, kv_heads * n_rep, seq_len, head_dim)
        )

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, past_length)

        if use_cache and self.cache is not None:
            self.cache.update(self.layer_idx, k, v)
            k_full, v_full = self.cache.get(self.layer_idx)
            total_seq_len = k_full.size(2)
        else:
            k_full, v_full = k, v
            total_seq_len = seq_len

        n_rep = self.n_heads // self.n_kv_heads
        k_rep = self._repeat_kv(k_full, n_rep)
        v_rep = self._repeat_kv(v_full, n_rep)

        attn_mask = None
        if self.sliding_window is not None and total_seq_len > self.sliding_window:
            q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
            k_idx = torch.arange(total_seq_len, device=x.device).unsqueeze(0)
            causal = (q_idx + past_length) >= k_idx
            window = (q_idx + past_length) - k_idx < self.sliding_window
            attn_mask = (causal & window)[None, None, :, :]

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attn_mask,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=attn_mask is None,
            )
        else:
            attn_weights = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
            else:
                mask = torch.triu(
                    torch.ones(seq_len, total_seq_len, dtype=torch.bool, device=x.device),
                    diagonal=1 + past_length,
                )
                attn_weights = attn_weights.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
            attn_weights = self.drop(attn_weights)
            attn_output = torch.matmul(attn_weights, v_rep)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attn_output)
