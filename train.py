#!/usr/bin/env python3
"""
MARJARA v2 – training entry point.

Single-GPU:
    python train.py --data data/train.bin --model_size small --mixed_precision

Multi-GPU (via torchrun):
    torchrun --nproc_per_node=4 train.py --data data/train.bin --distributed --mixed_precision
"""
import argparse
import random

import torch

from marjara import Learner


def parse_args():
    p = argparse.ArgumentParser(description="MARJARA v2 – LLM training")

    # Data
    p.add_argument("--data", type=str, required=True, help="Path to .txt or .bin tokenised file")
    p.add_argument("--val_data", type=str, default=None, help="Optional separate validation .bin file")
    p.add_argument("--tokenizer", type=str, default="cl100k_base", help="tiktoken encoding name")
    p.add_argument("--output_dir", type=str, default="checkpoints")

    # Model
    p.add_argument("--model_size", type=str, default="small", choices=["tiny", "small", "medium", "large"])
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--use_moe", action="store_true", help="Enable Mixture-of-Experts FFN")
    p.add_argument("--sliding_window", type=int, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Optimisation
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # Distributed
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--local_rank", type=int, default=0, help="Set automatically by torchrun")

    # Checkpointing
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--save_every", type=int, default=5, help="Save epoch checkpoint every N epochs")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # TF32
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    trainer = Learner(args)
    trainer.train()


if __name__ == "__main__":
    main()
