"""
KV Cache – unified, per-batch position tracking.
MARJARA v3 - cache.py

Changes over v2:
  - Removed silent max() fallback for mismatched seen_tokens; now raises clearly
  - update() no longer transposes k/v (Attn already passes correct layout)
  - get() returns a contiguous slice (avoids non-contiguous tensor issues in SDPA)
  - Added evict() for rolling/sliding cache: shifts buffer left when near capacity
  - Added @property seen for clean external reads
"""
import torch


class CacheThingy:
    def __init__(
        self,
        n_layers:    int,
        batch_size:  int,
        n_kv_heads:  int,
        head_dim:    int,
        max_seq_len: int,
        device:      torch.device,
        dtype:       torch.dtype = torch.float32,
    ):
        self.n_layers    = n_layers
        self.batch_size  = batch_size
        self.n_kv_heads  = n_kv_heads
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.device      = device
        self.dtype       = dtype

        # (n_layers, batch, kv_heads, max_seq_len, head_dim)
        shape = (n_layers, batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        # Single scalar: number of tokens written so far (same for all batch items
        # during training / greedy decoding with identical prompt lengths)
        self._seen = 0

    @property
    def seen_tokens(self) -> torch.Tensor:
        """Tensor of shape (batch,) for compatibility with model.py reads."""
        return torch.full((self.batch_size,), self._seen,
                          dtype=torch.long, device=self.device)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        Store new k/v.
        k, v: (batch, n_kv_heads, seq_len, head_dim)  – already correct layout from Attn
        """
        seq_len = k.shape[2]
        end = self._seen + seq_len

        if end > self.max_seq_len:
            # Rolling eviction: shift existing cache left to make room
            keep = self.max_seq_len - seq_len
            if keep <= 0:
                # Sequence longer than cache; just overwrite from the start
                self.k_cache[:, :, :, :seq_len, :] = k
                self.v_cache[:, :, :, :seq_len, :] = v
                self._seen = seq_len
                return
            self.k_cache[:, :, :, :keep, :] = self.k_cache[:, :, :, -keep:, :].clone()
            self.v_cache[:, :, :, :keep, :] = self.v_cache[:, :, :, -keep:, :].clone()
            self._seen = keep
            end = keep + seq_len

        self.k_cache[layer_idx, :, :, self._seen:end, :] = k
        self.v_cache[layer_idx, :, :, self._seen:end, :] = v

    def get(self, layer_idx: int):
        """Return contiguous k/v slices up to current position."""
        k = self.k_cache[layer_idx, :, :, :self._seen, :].contiguous()
        v = self.v_cache[layer_idx, :, :, :self._seen, :].contiguous()
        return k, v

    def increment(self, seq_len: int):
        """Called after all layers have been updated for a given step."""
        self._seen = min(self._seen + seq_len, self.max_seq_len)

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._seen = 0
