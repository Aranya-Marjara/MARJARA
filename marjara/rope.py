"""
Rotary Positional Embeddings (RoPE)
"""
import torch
import torch.nn as nn


class RotaryPos(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)        # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, past_length: int):
        seq_len = q.size(2)
        total_len = seq_len + past_length
        if total_len > self.cos_cached.shape[2]:
            self._build_cache(total_len)
        cos = self.cos_cached[:, :, past_length:total_len, :]
        sin = self.sin_cached[:, :, past_length:total_len, :]
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
