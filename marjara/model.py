"""
MARJARA v3 – Transformer Block and MarjaraModel.
model.py

Changes over v2:
  - Weight tying happens after _init_weights (bug fix, DeepSeek had this correct; kept)
  - Repetition penalty fully vectorised (no inner Python loop)
  - _sample() extracted cleanly; top-p uses a cleaner cumsum approach
  - generate() passes float mask instead of bool for SDPA compatibility
  - setup_cache() dtype defaults to float32 (avoids precision issues in generation under bf16 training)
  - clear_cache() sets seen to 0 on the object, not just None reference
  - MoE aux_loss accumulation uses += properly (avoids 0.0 + tensor type mixing)
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import HParams
from .attention import Attn, SimpleNorm
from .ffn import SwiGLU, MoELayer
from .cache import CacheThingy


class Block(nn.Module):
    def __init__(self, cfg: HParams, layer_idx: int):
        super().__init__()
        self.norm1 = SimpleNorm(cfg.d_model)
        self.norm2 = SimpleNorm(cfg.d_model)
        self.attn  = Attn(cfg, layer_idx)
        self.ffn   = MoELayer(cfg) if cfg.use_moe else SwiGLU(cfg)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.use_gc = cfg.gradient_checkpointing

    def _forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0):
        x = x + self.drop1(self.attn(self.norm1(x), use_cache=use_cache, past_length=past_length))
        residual = x
        normed   = self.norm2(x)
        if isinstance(self.ffn, MoELayer):
            ffn_out, losses = self.ffn(normed)
        else:
            ffn_out, losses = self.ffn(normed), {}
        x = residual + self.drop2(ffn_out)
        return x, losses

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0):
        if self.use_gc and self.training:
            return checkpoint(self._forward, x, use_cache, past_length, use_reentrant=False)
        return self._forward(x, use_cache, past_length)


class MarjaraModel(nn.Module):
    def __init__(self, cfg: HParams):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop        = nn.Dropout(cfg.dropout)
        self.blocks      = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm        = SimpleNorm(cfg.d_model)
        self.head        = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Init first, then tie – prevents head weights being overwritten
        self._init_weights()
        if cfg.weight_tying:
            self.head.weight = self.token_embed.weight

        self.kv_cache: Optional[CacheThingy] = None

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_out = any(k in name for k in ("out_proj", "w2", "down"))
                std = (
                    self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)
                    if is_out and self.cfg.scale_init_by_depth
                    else self.cfg.init_std
                )
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # ── KV cache helpers ────────────────────────────────────────────────────
    def setup_cache(self, batch_size: int, dtype: torch.dtype = torch.float32) -> CacheThingy:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        device   = next(self.parameters()).device
        self.kv_cache = CacheThingy(
            n_layers    = self.cfg.n_layers,
            batch_size  = batch_size,
            n_kv_heads  = self.cfg.n_kv_heads,
            head_dim    = head_dim,
            max_seq_len = self.cfg.context_length * 2,
            device      = device,
            dtype       = dtype,
        )
        for block in self.blocks:
            block.attn.cache = self.kv_cache
        return self.kv_cache

    def clear_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.reset()
        self.kv_cache = None
        for block in self.blocks:
            block.attn.cache = None

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, tokens: torch.Tensor, use_cache: bool = False):
        B, S = tokens.shape
        if S > self.cfg.context_length:
            tokens = tokens[:, -self.cfg.context_length:]
            S = self.cfg.context_length

        x = self.drop(self.token_embed(tokens))

        past_length = self.kv_cache._seen if (use_cache and self.kv_cache is not None) else 0

        aux_losses: Dict[str, torch.Tensor] = {}
        for block in self.blocks:
            x, losses = block(x, use_cache=use_cache, past_length=past_length)
            for k, v in losses.items():
                # Use proper tensor accumulation; avoids 0.0 + tensor issues
                aux_losses[k] = aux_losses[k] + v if k in aux_losses else v

        logits = self.head(self.norm(x))

        if use_cache and self.kv_cache is not None:
            self.kv_cache.increment(S)

        return (logits, aux_losses) if aux_losses else logits

    # ── Generation ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        prompt:             torch.Tensor,
        max_new_tokens:     int            = 100,
        temperature:        float          = 0.8,
        top_k:              Optional[int]  = 50,
        top_p:              Optional[float] = 0.9,
        repetition_penalty: float          = 1.0,
        stop_tokens:        Optional[List[int]] = None,
    ) -> torch.Tensor:
        self.eval()
        B = prompt.shape[0]
        self.setup_cache(B)

        # Prefill
        _ = self(prompt, use_cache=True)

        generated = prompt.clone()
        stop_set  = set(stop_tokens) if stop_tokens else None
        window    = 64

        for _ in range(max_new_tokens):
            if self.kv_cache._seen >= self.cfg.context_length * 2:
                break

            out    = self(generated[:, -1:], use_cache=True)
            logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :]   # (B, V)

            if repetition_penalty != 1.0:
                logits = self._apply_rep_penalty(logits, generated[:, -window:], repetition_penalty)

            logits     = logits / max(temperature, 1e-6)
            next_token = self._sample(logits, top_k, top_p)               # (B, 1)
            generated  = torch.cat([generated, next_token], dim=1)

            if stop_set and next_token[0, 0].item() in stop_set:
                break

        self.clear_cache()
        return generated

    @staticmethod
    def _apply_rep_penalty(
        logits:  torch.Tensor,    # (B, V)
        recent:  torch.Tensor,    # (B, W)
        penalty: float,
    ) -> torch.Tensor:
        """Fully vectorised repetition penalty – no Python loops."""
        B, V = logits.shape
        # Build occurrence mask: (B, V) True where token appeared recently
        mask = torch.zeros(B, V, dtype=torch.bool, device=logits.device)
        mask.scatter_(1, recent, True)
        # Apply penalty where mask is True
        logits = logits.clone()
        logits[mask] = logits[mask] / penalty
        return logits

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        top_k:  Optional[int],
        top_p:  Optional[float],
    ) -> torch.Tensor:
        """Top-k and top-p (nucleus) sampling."""
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = logits.topk(k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Shift right so the token that pushes us over top_p is kept
            remove = (cumprobs - sorted_logits.softmax(dim=-1)) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            # Scatter back to original ordering
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
