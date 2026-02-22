"""
MARJARA v2 – Transformer Block and main MarjaraModel.
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
        self.attn = Attn(cfg, layer_idx)
        self.ffn = MoELayer(cfg) if cfg.use_moe else SwiGLU(cfg)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.gradient_checkpointing = cfg.gradient_checkpointing

    def _forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0):
        x = x + self.drop1(self.attn(self.norm1(x), use_cache=use_cache, past_length=past_length))
        residual = x
        x = self.norm2(x)
        if isinstance(self.ffn, MoELayer):
            x, losses = self.ffn(x)
        else:
            x = self.ffn(x)
            losses = {}
        x = residual + self.drop2(x)
        return x, losses

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_length: int = 0):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward, x, use_cache, past_length)
        return self._forward(x, use_cache, past_length)


class MarjaraModel(nn.Module):
    def __init__(self, cfg: HParams):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = SimpleNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.head.weight = self.token_embed.weight

        self._init_weights()
        self.kv_cache: Optional[CacheThingy] = None

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_out = "out_proj" in name or "w2" in name
                std = (
                    self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)
                    if is_out and self.cfg.scale_init_by_depth
                    else self.cfg.init_std
                )
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------
    def setup_cache(self, batch_size: int) -> CacheThingy:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.kv_cache = CacheThingy(
            n_layers=self.cfg.n_layers,
            batch_size=batch_size,
            n_kv_heads=self.cfg.n_kv_heads,
            head_dim=head_dim,
            max_seq_len=self.cfg.context_length * 2,
            device=device,
            dtype=dtype,
        )
        for block in self.blocks:
            block.attn.cache = self.kv_cache
        return self.kv_cache

    def clear_cache(self):
        self.kv_cache = None
        for block in self.blocks:
            block.attn.cache = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor, use_cache: bool = False):
        batch, seq_len = tokens.shape
        if seq_len > self.cfg.context_length:
            tokens = tokens[:, -self.cfg.context_length:]
            seq_len = self.cfg.context_length

        x = self.drop(self.token_embed(tokens))

        past_length = int(self.kv_cache.seen_tokens[0]) if use_cache and self.kv_cache is not None else 0

        aux_losses: Dict[str, torch.Tensor] = {}
        for block in self.blocks:
            x, losses = block(x, use_cache=use_cache, past_length=past_length)
            for k, v in losses.items():
                aux_losses[k] = aux_losses.get(k, 0.0) + v

        logits = self.head(self.norm(x))

        if use_cache and self.kv_cache is not None:
            self.kv_cache.increment(seq_len)

        return (logits, aux_losses) if aux_losses else logits

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
    ):
        self.eval()
        batch_size = prompt.shape[0]
        cache = self.setup_cache(batch_size)

        # Prefill
        _ = self(prompt, use_cache=True)

        generated = prompt.clone()
        stop_set = set(stop_tokens) if stop_tokens else None
        recent_window = 64

        for _ in range(max_new_tokens):
            if cache.seen_tokens[0] >= self.cfg.context_length * 2:
                break

            out = self(generated[:, -1:], use_cache=True)
            logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :]

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token in set(generated[i, -recent_window:].tolist()):
                        logits[i, token] /= repetition_penalty

            logits = logits / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, remove)
                logits[indices_to_remove] = float("-inf")

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if stop_set and next_token[0, 0].item() in stop_set:
                break

        self.clear_cache()
        return generated
