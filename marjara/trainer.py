"""
MARJARA v3 – Learner (Trainer)
trainer.py

Changes over v2:
  - DDP val loss sync corrected: uses all_reduce on tensors (not .item() before reduce)
  - has_grads flag properly prevents spurious optimizer step on empty leftovers
  - load_checkpoint: weights_only defaults to True for security; caller opts out explicitly
  - DataLoader uses persistent_workers + prefetch_factor (DeepSeek had this; kept + fixed for num_workers=0 edge case)
  - MoE noise decay: linear schedule, applied correctly only to MoELayer modules
  - TensorBoard + W&B integrated cleanly with None-guards
  - Early stopping: patience-based with configurable min_delta
  - train_epoch logs lr per step (not just epoch-end)
  - validate() returns avg_ce that is rank-0 consistent across all DDP ranks
  - _update_ema() uses lerp_ for in-place EMA (faster than mul_+add_)
  - Removed pre-allocated logits_cache buffer (was unused in inner loop)
"""
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import tiktoken

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

try:
    import wandb as _wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from .config import HParams
from .model import MarjaraModel
from .datasets import TextDataset, MMapSliceDataset
from .ffn import MoELayer


class Learner:
    def __init__(self, args):
        self.args = args
        self._setup_logging()
        self._setup_distributed()
        self._setup_device()
        self._setup_tokenizer()
        self._load_data()
        self._build_model()
        self._setup_ema()
        self._maybe_compile()
        self._maybe_ddp()
        self._setup_optimizer_scheduler()
        self._setup_mixed_precision()
        self._setup_checkpointing()
        self._setup_log_backends()

        self.current_step        = 0
        self.best_val_ce         = float("inf")
        self.epochs_no_improve   = 0
        self.has_grads           = False

    # ── Setup helpers -----------------------------------------------------------

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO if getattr(self.args, "local_rank", 0) == 0 else logging.WARNING,
        )
        self.logger = logging.getLogger(__name__)

    def _setup_distributed(self):
        self.is_dist   = self.args.distributed
        self.local_rank = self.args.local_rank
        self.world_size = 1
        self.rank       = 0
        if self.is_dist:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
            self.rank       = dist.get_rank()
            torch.cuda.set_device(self.local_rank)

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            if self.is_dist:
                self.logger.warning("Distributed on CPU unsupported; disabling DDP.")
                self.is_dist = False

    def _setup_tokenizer(self):
        try:
            self.tokenizer = tiktoken.get_encoding(self.args.tokenizer)
        except Exception:
            self.logger.warning("Tokenizer not found; falling back to cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _load_data(self):
        if self.args.data.endswith(".bin"):
            self.logger.info(f"Loading mmap dataset: {self.args.data}")
            mm = np.memmap(self.args.data, dtype=np.uint16, mode="r")
            total = len(mm)
            self.args.vocab_size = max(50304, int(mm.max()) + 1)
            split = int(0.9 * total)
            if self.args.val_data:
                vm = np.memmap(self.args.val_data, dtype=np.uint16, mode="r")
                self.train_dataset = MMapSliceDataset(self.args.data, 0, total, self.args.context_length)
                self.val_dataset   = MMapSliceDataset(self.args.val_data, 0, len(vm), self.args.context_length)
            else:
                self.train_dataset = MMapSliceDataset(self.args.data, 0, split, self.args.context_length)
                self.val_dataset   = MMapSliceDataset(self.args.data, split, total, self.args.context_length)
        else:
            self.logger.info(f"Loading text file: {self.args.data}")
            with open(self.args.data, encoding="utf-8") as f:
                text = f.read()
            toks = self.tokenizer.encode(text)
            self.args.vocab_size = self.tokenizer.n_vocab
            data  = torch.tensor(toks, dtype=torch.long)
            split = int(0.9 * len(data))
            self.train_dataset = TextDataset(data[:split], self.args.context_length)
            self.val_dataset   = TextDataset(data[split:], self.args.context_length)

        sampler = None
        shuffle = True
        if self.is_dist:
            sampler = DistributedSampler(
                self.train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True, drop_last=True,
            )
            shuffle = False
        self.train_sampler = sampler

        nw = min(4, os.cpu_count() or 1)
        # prefetch_factor only valid when num_workers > 0
        pf = 2 if nw > 0 else None

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size,
            shuffle=shuffle, sampler=sampler,
            num_workers=nw, pin_memory=True, drop_last=True,
            persistent_workers=(nw > 0), prefetch_factor=pf,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=nw, pin_memory=True,
            persistent_workers=(nw > 0), prefetch_factor=pf,
        ) if self.val_dataset else None

        self.logger.info(f"Train: {len(self.train_dataset)}  Val: {len(self.val_dataset) if self.val_dataset else 0}")

    def _build_model(self):
        preset = {"tiny": HParams.tiny, "small": HParams.small,
                  "medium": HParams.medium, "large": HParams.large}
        cfg = preset[self.args.model_size]() if self.args.model_size in preset else HParams()
        cfg.vocab_size            = self.args.vocab_size
        cfg.context_length        = self.args.context_length
        cfg.use_moe               = self.args.use_moe
        cfg.sliding_window        = self.args.sliding_window
        cfg.gradient_checkpointing = self.args.gradient_checkpointing
        self.cfg   = cfg
        self.model = MarjaraModel(cfg).to(self.device)
        self.logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def _setup_ema(self):
        self.ema_model = None
        if self.args.use_ema:
            self.ema_model = MarjaraModel(self.cfg).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_decay = self.args.ema_decay
            self.logger.info(f"EMA decay={self.ema_decay}")

    def _maybe_compile(self):
        if self.args.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with torch.compile")

    def _maybe_ddp(self):
        if self.is_dist:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def _setup_optimizer_scheduler(self):
        decay, no_decay, seen = [], [], set()
        for name, p in self.model.named_parameters():
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            if p.ndim < 2 or any(k in name.lower() for k in ("norm", "embed", "bias")):
                no_decay.append(p)
            else:
                decay.append(p)

        groups = [
            {"params": decay,    "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        try:
            self.optimizer = AdamW(groups, lr=self.args.lr, betas=(0.9, 0.95), fused=True)
            self.logger.info("Fused AdamW")
        except TypeError:
            self.optimizer = AdamW(groups, lr=self.args.lr, betas=(0.9, 0.95))
            self.logger.info("Standard AdamW")

        spe        = len(self.train_loader)
        eff        = math.ceil(spe / self.args.accumulation_steps)
        total      = self.args.epochs * eff
        warmup_end = self.args.warmup_steps

        warmup  = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_end)
        cosine  = CosineAnnealingLR(self.optimizer, T_max=max(1, total - warmup_end), eta_min=1e-5)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_end])

    def _setup_mixed_precision(self):
        self.use_amp = self.args.mixed_precision and self.device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.logger.info(f"AMP: {self.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler("cuda") if self.use_amp else None
        except ImportError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _setup_checkpointing(self):
        self.ckpt_dir = Path(self.args.output_dir)
        self.ckpt_dir.mkdir(exist_ok=True)

    def _setup_log_backends(self):
        self.tb_writer = None
        if self.rank == 0:
            if _TB:
                self.tb_writer = SummaryWriter(log_dir=str(self.ckpt_dir / "tb_logs"))
                self.logger.info("TensorBoard enabled")
            if _WANDB and getattr(self.args, "wandb_project", None):
                _wandb.init(project=self.args.wandb_project, config=vars(self.args))
                self.logger.info("W&B enabled")

    # ── Utilities -----------------------------------------------------------------------------

    def _log(self, metrics: Dict, step: int):
        """Log a dict of scalar metrics to all enabled backends."""
        if self.rank != 0:
            return
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
        if _WANDB and _wandb.run:
            _wandb.log(metrics, step=step)

    def _update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            src = {k.replace("module.", ""): v
                   for k, v in self.model.state_dict().items()}
            dst = self.ema_model.state_dict()
            for name, ema_p in dst.items():
                if name in src:
                    # lerp_(a, b, weight) performs: ema = (1-d)*ema + d*src  in-place
                    ema_p.lerp_(src[name].to(ema_p.dtype), 1.0 - self.ema_decay)

    def _decay_moe_noise(self, epoch: int, start: int):
        """Linearly decay MoE routing noise to zero over first 80% of training."""
        if not self.cfg.use_moe:
            return
        total    = self.args.epochs - start
        progress = (epoch - start) / max(1, total)
        noise    = self.cfg.moe_noise_std * max(0.0, 1.0 - progress / 0.8)
        raw = self.model.module if self.is_dist else self.model
        for m in raw.modules():
            if isinstance(m, MoELayer):
                m.set_noise_std(noise)

    def _compute_loss(self, logits, targets, aux_losses=None) -> Tuple[torch.Tensor, torch.Tensor]:
        ce   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss = ce
        if aux_losses:
            for v in aux_losses.values():
                if isinstance(v, torch.Tensor):
                    loss = loss + v
        return loss, ce

    def _optimizer_step(self):
        if self.use_amp and self.scaler:
            self.scaler.unscale_(self.optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        if self.rank == 0 and self.current_step % 100 == 0:
            self._log({"train/grad_norm": gnorm.item()}, self.current_step)
        if self.use_amp and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)   # set_to_none=True: less memory than zero
        self._update_ema()
        self.has_grads = False

    # ── Train / Validate ----------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Tuple[float, float, Dict]:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        total_loss = total_ce = 0.0
        total_aux  = {}
        n = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.has_grads = False

        log_every = max(1, len(self.train_loader) // 100)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=(self.rank != 0))

        for i, (x, y) in enumerate(pbar):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(x)
                logits, aux = out if isinstance(out, tuple) else (out, None)
                loss, ce    = self._compute_loss(logits, y, aux)

            (loss / self.args.accumulation_steps).backward() if not (self.use_amp and self.scaler) \
                else self.scaler.scale(loss / self.args.accumulation_steps).backward()
            self.has_grads = True

            if (i + 1) % self.args.accumulation_steps == 0:
                self._optimizer_step()
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                self._log({"train/loss_step": loss.item(), "train/lr": lr}, self.current_step)
                self.current_step += 1

            total_loss += loss.item()
            total_ce   += ce.item()
            if aux:
                for k, v in aux.items():
                    total_aux[k] = total_aux.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)
            n += 1

            if self.rank == 0 and (i + 1) % log_every == 0:
                pf = {"loss": f"{loss.item():.4f}", "ce": f"{ce.item():.4f}"}
                if aux:
                    pf.update({k: f"{v.item():.4f}" for k, v in aux.items() if isinstance(v, torch.Tensor)})
                pbar.set_postfix(pf)

        # Flush any leftover gradient accumulation
        if self.has_grads:
            self._optimizer_step()
            self.scheduler.step()
            self.current_step += 1

        avg_aux = {k: v / n for k, v in total_aux.items()}
        return total_loss / n, total_ce / n, avg_aux

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        if self.val_loader is None:
            return 0.0, 0.0
        self.model.eval()
        total_loss = total_ce = n = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(x)
                logits, aux = out if isinstance(out, tuple) else (out, None)
                loss, ce    = self._compute_loss(logits, y, aux)
            total_loss += loss.item()
            total_ce   += ce.item()
            n          += 1

        # Synchronise across DDP ranks BEFORE computing averages
        if self.is_dist:
            buf = torch.tensor([total_loss, total_ce, float(n)], device=self.device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            total_loss, total_ce, n = buf[0].item(), buf[1].item(), int(buf[2].item())

        avg_ce  = total_ce   / max(n, 1)
        avg_loss = total_loss / max(n, 1)
        ppl     = math.exp(min(avg_ce, 20.0))

        if self.rank == 0:
            self.logger.info(f"Val | CE={avg_ce:.4f} PPL={ppl:.2f}")
            self._log({"val/ce": avg_ce, "val/ppl": ppl}, self.current_step)
        return avg_loss, avg_ce

    # ── Checkpointing -----------------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, val_ce: float, is_best: bool):
        if self.rank != 0:
            return
        raw = self.model.module if self.is_dist else self.model
        ckpt = {
            "epoch":              epoch,
            "model_state_dict":   raw.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":  self.scaler.state_dict() if self.scaler else None,
            "val_loss":           val_loss,
            "val_ce":             val_ce,
            "config":             asdict(self.cfg),
            "tokenizer_name":     self.args.tokenizer,
            "args":               vars(self.args),
            "step":               self.current_step,
        }
        if self.ema_model:
            ckpt["ema_state_dict"] = self.ema_model.state_dict()

        torch.save(ckpt, self.ckpt_dir / "checkpoint_latest.pt")
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "checkpoint_best.pt")
        if epoch % self.args.save_every == 0:
            torch.save(ckpt, self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt")
        self.logger.info(f"Checkpoint saved: epoch={epoch}")

    def load_checkpoint(self, path: Union[str, Path], weights_only: bool = True):
        """
        weights_only=True (default) is safe for untrusted checkpoints.
        Pass weights_only=False only for checkpoints you generated yourself.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=weights_only)
        raw  = self.model.module if self.is_dist else self.model
        raw.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.ema_model and "ema_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state_dict"])
        self.current_step = ckpt.get("step", 0)
        return ckpt["epoch"], ckpt.get("val_ce", float("inf"))

    # ── Main loop --------------------------------------------------------------

    @torch.no_grad()
    def generate_sample(self, prompt: str, max_tokens: int = 100) -> str:
        model = self.ema_model or (self.model.module if self.is_dist else self.model)
        model.eval()
        ids  = self.tokenizer.encode(prompt)
        t    = torch.tensor([ids], device=self.device)
        out  = model.generate(t, max_new_tokens=max_tokens, temperature=0.8,
                               top_k=50, top_p=0.9, repetition_penalty=1.1)
        return self.tokenizer.decode(out[0].tolist())

    def train(self):
        start = 0
        if self.args.resume:
            start, _ = self.load_checkpoint(self.args.resume, weights_only=False)
            self.logger.info(f"Resumed from epoch {start}")

        patience  = getattr(self.args, "early_stop_patience",   5)
        min_delta = getattr(self.args, "early_stop_min_delta", 0.0)

        for epoch in range(start, self.args.epochs):
            self._decay_moe_noise(epoch, start)

            train_loss, train_ce, aux = self.train_epoch(epoch)
            val_loss,   val_ce        = self.validate()

            is_best = val_ce < self.best_val_ce - min_delta
            if is_best:
                self.best_val_ce       = val_ce
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            lr  = self.scheduler.get_last_lr()[0]
            msg = (f"Ep {epoch+1}/{self.args.epochs} | "
                   f"train_loss={train_loss:.4f} train_ce={train_ce:.4f} "
                   f"val_ce={val_ce:.4f} lr={lr:.2e}"
                   + (" ★BEST" if is_best else ""))
            if aux:
                msg += " | " + " ".join(f"{k}={v:.4f}" for k, v in aux.items())
            self.logger.info(msg)

            self._log({"train/loss": train_loss, "train/ce": train_ce,
                       "val/ce": val_ce, "lr": lr}, epoch)

            if (epoch + 1) % 5 == 0 and self.rank == 0:
                sample = self.generate_sample("The future of AI is")
                self.logger.info(f"Sample: {sample}")
                if self.tb_writer:
                    self.tb_writer.add_text("sample", sample, epoch)
                if _WANDB and _wandb.run:
                    _wandb.log({"sample": sample}, step=epoch)

            self.save_checkpoint(epoch + 1, val_loss, val_ce, is_best)

            if self.epochs_no_improve >= patience:
                self.logger.info(f"Early stopping after {patience} epochs without improvement.")
                break

        if self.tb_writer:
            self.tb_writer.close()
        if self.is_dist:
            dist.destroy_process_group()
