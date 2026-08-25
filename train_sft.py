#!/usr/bin/env python3
"""
train_sft.py — Stage 2: Supervised Fine-Tuning (Instruction Tuning).

This script takes a pre-trained model and teaches it to act like a chatbot.
It uses an instruction-following dataset (Alpaca) and applies LOSS MASKING,
which means the model is only penalized for mistakes it makes while generating
the Assistant's response. It is not penalized for the User's prompt.

Usage:
    # First, you must have a pre-trained checkpoint from train_real.py
    python3 train_sft.py --config 300m --load-ckpt ckpt_final.pt --batch-size 1 --grad-accum 16
"""

from __future__ import annotations
import argparse
import sys
import time
import math
import os
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("\n[ERROR] Missing required packages.")
    print("Please install them using: pip install datasets transformers\n")
    sys.exit(1)

sys.path.insert(0, ".")
from src import Transformer, DEFAULT_300M, TINY_TEST


# ── Configuration ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config",       default="300m", choices=["tiny", "300m"])
    p.add_argument("--device",       default="auto")
    p.add_argument("--load-ckpt",    type=str,   required=True, help="Path to pre-trained Stage 1 checkpoint")
    p.add_argument("--steps",        type=int,   default=10_000, help="SFT steps (usually much shorter than pre-training)")
    p.add_argument("--batch-size",   type=int,   default=1, help="Micro-batch size")
    p.add_argument("--grad-accum",   type=int,   default=16, help="Gradient accumulation steps")
    p.add_argument("--lr",           type=float, default=2e-5, help="SFT learning rate (much lower than pre-training)")
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--save-every",   type=int,   default=1000)
    return p.parse_args()


# ── Data Pipeline with LOSS MASKING ───────────────────────────────────────────

def stream_sft_batches(
    tokenizer: AutoTokenizer,
    batch_size: int,
    block_size: int,
):
    """
    Streams the Alpaca dataset, formats it into a chat template, and
    creates masked targets (setting user prompt tokens to -100 so they
    are ignored in the cross-entropy loss).
    """
    # Load instruction dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    
    # We use simple text tags since Mistral base tokenizer doesn't have special chat tokens
    PROMPT_PREFIX = "USER: "
    RESPONSE_PREFIX = "\nASSISTANT: "
    EOS = tokenizer.eos_token or "</s>"
    
    # To find where the response starts during tokenization, we tokenize the prefix
    # Note: tokenizers can behave weirdly with prefixes, so we do a simple string search.
    
    batch_input_ids = []
    batch_targets = []
    
    for row in dataset:
        # Build the conversation string
        instruction = row["instruction"]
        if row.get("input", ""):
            instruction += "\n" + row["input"]
        
        response = row["output"]
        
        full_text = f"{PROMPT_PREFIX}{instruction}{RESPONSE_PREFIX}{response}{EOS}"
        
        # Tokenize the full conversation
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        
        # We need to find where the response starts so we can mask the prompt.
        # A robust way is to tokenize the prompt by itself and find its length.
        prompt_text = f"{PROMPT_PREFIX}{instruction}{RESPONSE_PREFIX}"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        response_start_idx = len(prompt_tokens)
        
        if response_start_idx >= block_size - 4:
            continue
        
        # Truncate if it exceeds block size
        tokens = tokens[:block_size]
        
        # Create targets: copy inputs, then mask everything before the response
        targets = list(tokens)
        for i in range(min(response_start_idx, len(targets))):
            targets[i] = -100  # -100 is ignored by PyTorch CrossEntropyLoss
            
        # Shift targets by 1 for language modeling (predicting the NEXT token)
        # We do this later in the collate logic or right here:
        input_ids = tokens[:-1]
        shifted_targets = targets[1:]
        
        # Pad to block_size (or just pad the batch). For simplicity, we pad to block_size
        pad_id = tokenizer.pad_token_id or 0
        while len(input_ids) < block_size:
            input_ids.append(pad_id)
            shifted_targets.append(-100)
            
        # Truncate strictly to block_size in case of off-by-one
        input_ids = input_ids[:block_size]
        shifted_targets = shifted_targets[:block_size]
            
        batch_input_ids.append(input_ids)
        batch_targets.append(shifted_targets)
        
        if len(batch_input_ids) == batch_size:
            yield {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "targets": torch.tensor(batch_targets, dtype=torch.long),
                "position_ids": torch.arange(block_size, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
                "attn_mask": None, # Standard causal mask
            }
            batch_input_ids = []
            batch_targets = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_moe_losses(model: torch.nn.Module) -> tuple[float, float]:
    aux_loss, z_loss = 0.0, 0.0
    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "l_aux"):
            v = mlp.l_aux
            aux_loss += v.item() if isinstance(v, torch.Tensor) else v
        if hasattr(mlp, "l_z"):
            v = mlp.l_z
            z_loss += v.item() if isinstance(v, torch.Tensor) else v
    return aux_loss, z_loss


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    device = args.device
    if device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dev = torch.device(device)

    config = TINY_TEST if args.config == "tiny" else DEFAULT_300M
    config.use_gradient_checkpointing = True

    print(f"\n[SFT] Loading model architecture...")
    model = Transformer(config).to(dev)
    
    print(f"[SFT] Loading pre-trained checkpoint: {args.load_ckpt}")
    if not os.path.exists(args.load_ckpt):
        print(f"[ERROR] Checkpoint {args.load_ckpt} not found!")
        sys.exit(1)
        
    ckpt = torch.load(args.load_ckpt, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    print("[SFT] Checkpoint loaded successfully. Model is ready to learn instructions.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    print("[SFT] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    print("[SFT] Connecting to SFT dataset (yahma/alpaca-cleaned)...")
    batch_iterator = stream_sft_batches(tokenizer, args.batch_size, config.block_size)

    use_amp = (dev.type in ("cuda", "mps"))
    amp_ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp) if use_amp else nullcontext()

    model.train()
    t_start = time.perf_counter()
    t_last = t_start
    
    moe_aux_coef = getattr(config, "moe_aux_loss_coef", 0.0)
    moe_z_coef   = getattr(config, "moe_z_loss_coef",   0.0)

    print("\nStarting Instruction Tuning (SFT)...")
    
    try:
        for step in range(1, args.steps + 1):
            
            optimizer.zero_grad(set_to_none=True)
            step_lm_loss = 0.0

            for _ in range(args.grad_accum):
                batch = next(batch_iterator)
                input_ids    = batch["input_ids"].to(dev, non_blocking=True)
                targets      = batch["targets"].to(dev, non_blocking=True)
                position_ids = batch["position_ids"].to(dev, non_blocking=True)
                
                with amp_ctx:
                    # Targets with -100 are automatically ignored in the loss calculation!
                    _, lm_loss = model(input_ids, position_ids=position_ids, targets=targets)
                    aux_loss, z_loss = collect_moe_losses(model)
                    total_loss = lm_loss + (moe_aux_coef * aux_loss) + (moe_z_coef * z_loss)
                    total_loss = total_loss / args.grad_accum
                
                total_loss.backward()
                step_lm_loss += lm_loss.item() / args.grad_accum

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                now = time.perf_counter()
                t_last = now
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  SFT step {step:5d} | mask_loss: {step_lm_loss:.4f} | lr: {current_lr:.2e}")

            if step % args.save_every == 0:
                path = f"sft_ckpt_step_{step}.pt"
                torch.save({"model_state_dict": model.state_dict()}, path)
                print(f"  [SAVE] Checkpoint saved: {path}")
                
    except KeyboardInterrupt:
        print("\n\n[WARNING] KeyboardInterrupt. Saving interrupted SFT checkpoint...")
        torch.save({"model_state_dict": model.state_dict()}, "sft_ckpt_interrupted.pt")
        sys.exit(0)

    print("\nSFT Complete.")
    torch.save({"model_state_dict": model.state_dict()}, "sft_ckpt_final.pt")

if __name__ == "__main__":
    main()
