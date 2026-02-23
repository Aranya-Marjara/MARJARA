"""
Rotary Positional Embeddings (RoPE) with NTK and linear scaling.
MARJARA v3 - rope.py

Changes over DeepSeek v2:
  - Cache rebuild doubles capacity each time (avoids rebuild on every new length)
  - NTK scaling applied to inv_freq (theta scaling), not to positional indices
  - Linear scaling applied to position indices (correct per Su et al.)
  - YaRN placeholder removed; only two well-understood modes kept
  - _build_cache moves to device of inv_freq automatically (handles multi-GPU correctly)
  - register_buffer used consistently; cos/sin not duplicated in emb
"""
import torch
import torch.nn as nn


class RotaryPos(nn.Module):
    """
    Rotary Position Embedding with optional length extrapolation scaling.

    scaling_type:
      "none"   – standard RoPE (default)
      "linear" – divide position by scale_factor (position interpolation)
      "ntk"    – adjust theta by scale_factor (NTK-aware scaling, better quality)
    """

    def __init__(
        self,
        dim:          int,
        max_seq_len:  int   = 4096,
        theta:        float = 10000.0,
        scaling_type: str   = "none",
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim          = dim
        self.scaling_type = scaling_type
        self.scale_factor = scale_factor

        # NTK: scale theta so higher frequencies extrapolate better
        effective_theta = theta * (scale_factor ** (dim / (dim - 2))) \
                          if scaling_type == "ntk" else theta

        inv_freq = 1.0 / (effective_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cache_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pre-compute cos/sin tables up to seq_len."""
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Linear scaling: compress positions into a smaller range
        if self.scaling_type == "linear":
            t = t / self.scale_factor

        freqs = torch.outer(t, self.inv_freq)           # (seq_len, dim//2)
        emb   = torch.cat([freqs, freqs], dim=-1)       # (seq_len, dim)

        # Shape: (1, 1, seq_len, dim)  ready for broadcasting over (batch, heads, seq, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)
        self._cache_len = seq_len

    def _ensure_cache(self, total_len: int):
        if total_len > self._cache_len:
            # Double until large enough – avoids rebuilding on every new length
            new_len = self._cache_len
            while new_len < total_len:
                new_len *= 2
            self._build_cache(new_len)

    def forward(self, q: torch.Tensor, k: torch.Tensor, past_length: int) -> tuple:
        """
        q, k: (batch, heads, seq_len, head_dim)
        Returns rotated (q, k) with the same shapes.
        """
        seq_len   = q.size(2)
        total_len = past_length + seq_len
        self._ensure_cache(total_len)

        cos = self.cos_cached[:, :, past_length:total_len, :]  # (1,1,seq,dim)
        sin = self.sin_cached[:, :, past_length:total_len, :]

        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and rotate: [x1, x2] -> [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
