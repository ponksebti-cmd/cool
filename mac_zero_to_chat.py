#!/usr/bin/env python3
"""
mac_zero_to_chat.py — All-in-One script specifically optimized for Apple Silicon (M1/M2/M3)

This script trains a tiny model on an instruction dataset and visually tracks progress,
then immediately drops you into an interactive chat loop with the model you just trained.

Optimizations for M2:
  - Device: 'mps' (Metal Performance Shaders)
  - Precision: bfloat16
  - Unified Memory conscious: Streams data to avoid CPU/GPU transfer bottlenecks.
"""

from __future__ import annotations
import sys
import time
import math
from contextlib import nullcontext
import traceback

import torch
import torch.optim as optim

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("\n[ERROR] Missing required packages.")
    print("Please install them using: pip install datasets transformers\n")
    sys.exit(1)

sys.path.insert(0, ".")
from src import Transformer, TINY_TEST

# ── Custom Terminal Progress Bar ──────────────────────────────────────────────
def print_progress(step: int, total: int, loss: float, t0: float, tokens: int, lr: float):
    elapsed = time.perf_counter() - t0
    tok_per_sec = tokens / elapsed if elapsed > 0 else 0
    percent = step / total
    
    # Calculate ETA
    if step > 0:
        total_time_est = elapsed / percent
        eta_seconds = total_time_est - elapsed
        m, s = divmod(int(eta_seconds), 60)
        eta_str = f"{m}m {s:02d}s"
    else:
        eta_str = "..."

    bar_len = 25
    filled_len = int(bar_len * percent)
    bar = '█' * filled_len + '░' * (bar_len - filled_len)
    
    sys.stdout.write(
        f"\r  [{bar}] {percent*100:3.0f}% | "
        f"Loss: {loss:.4f} | "
        f"Tok/s: {tok_per_sec:,.0f} | "
        f"LR: {lr:.2e} | "
        f"ETA: {eta_str} "
    )
    sys.stdout.flush()

# ── Data Streaming ────────────────────────────────────────────────────────────
def stream_alpaca(tokenizer: AutoTokenizer, batch_size: int, block_size: int):
    # We use a tiny subset of the chat dataset for quick local training
    dataset = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    
    batch_inputs, batch_targets = [], []
    for row in dataset:
        prompt = f"USER: {row['instruction']}\nASSISTANT: "
        if row.get('input', ''):
            prompt = f"USER: {row['instruction']}\n{row['input']}\nASSISTANT: "
            
        full_text = prompt + row['output'] + (tokenizer.eos_token or "</s>")
        
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        
        # If the prompt is longer than the block size, the entire sequence will
        # be masked as -100, causing a NaN loss. Skip these long examples.
        if prompt_len >= block_size - 4:
            continue
            
        tokens = tokens[:block_size]
        targets = list(tokens)
        
        # Mask user prompt
        for i in range(min(prompt_len, len(targets))):
            targets[i] = -100
            
        input_ids = tokens[:-1]
        shifted_targets = targets[1:]
        
        pad_id = tokenizer.pad_token_id or 0
        while len(input_ids) < block_size:
            input_ids.append(pad_id)
            shifted_targets.append(-100)
            
        batch_inputs.append(input_ids[:block_size])
        batch_targets.append(shifted_targets[:block_size])
        
        if len(batch_inputs) == batch_size:
            yield {
                "input_ids": torch.tensor(batch_inputs, dtype=torch.long),
                "targets": torch.tensor(batch_targets, dtype=torch.long),
                "position_ids": torch.arange(block_size, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
                "attn_mask": None
            }
            batch_inputs, batch_targets = [], []

# ── Generation ────────────────────────────────────────────────────────────────
def chat_loop(model: Transformer, tokenizer: AutoTokenizer, dev: torch.device):
    print("\n\n" + "═"*60)
    print("  🧠 TRAINING COMPLETE. AI ASSISTANT IS READY.")
    print("     (Type 'quit' to exit)")
    print("═"*60 + "\n")
    
    while True:
        try:
            user_input = input("\033[96mYou:\033[0m ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue
                
            prompt = f"USER: {user_input}\nASSISTANT: "
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
            
            print("\033[92mAssistant:\033[0m ", end="", flush=True)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_tensor,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_k=40
                )
                
            new_tokens = output_ids[0, len(prompt_ids):].tolist()
            if tokenizer.eos_token_id in new_tokens:
                new_tokens = new_tokens[:new_tokens.index(tokenizer.eos_token_id)]
                
            print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip() + "\n")
            
        except KeyboardInterrupt:
            break

# ── Main Script ───────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  MacBook M2 Zero-to-Chat Pipeline")
    print(f"{'='*60}")

    if not torch.backends.mps.is_available():
        print("[WARNING] MPS not detected! This script is meant for Apple Silicon.")
        dev = torch.device("cpu")
    else:
        dev = torch.device("mps")
        print(f"  Hardware : Apple Silicon (MPS)")
        
    print(f"  Precision: bfloat16")
    print(f"  Config   : TINY (Fast local iteration)")
    print(f"{'='*60}\n")
    
    # 1. Setup
    config = TINY_TEST
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Must override vocab size because TINY_TEST defaults to 256 (byte-level), 
    # but the Mistral tokenizer has ~32,000 tokens. On MPS, out-of-bounds 
    # embedding lookups silently return NaNs instead of throwing IndexErrors!
    config.vocab_size = tokenizer.vocab_size
    config.use_gradient_checkpointing = True
    
    # Since it's a tiny model, let's keep the experimental smartness features on
    # but dial down pondering slightly so it doesn't take forever on a laptop.
    config.ponder_steps = 1
    config.use_critic_expert = True
    config.use_memory_workspace = True

    model = Transformer(config).to(dev)
    
    # 2. Optimizer & Data
    lr = 5e-4
    steps = 200  # Extremely short run just for interactive demonstration
    batch_size = 4
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    
    batch_iterator = stream_alpaca(tokenizer, batch_size, config.block_size)
    amp_ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16) if dev.type == "mps" else nullcontext()

    # 3. Training Loop
    print("\n🚀 Starting Training Phase...\n")
    model.train()
    
    tokens_seen = 0
    t0 = time.perf_counter()
    
    try:
        for step in range(1, steps + 1):
            batch = next(batch_iterator)
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            targets = batch["targets"].to(dev, non_blocking=True)
            pos_ids = batch["position_ids"].to(dev, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with amp_ctx:
                _, loss = model(input_ids, position_ids=pos_ids, targets=targets)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            tokens_seen += input_ids.numel()
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Update progress bar
            print_progress(step, steps, loss.item(), t0, tokens_seen, current_lr)
            
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted early. Entering chat with partial brain...")
    except Exception as e:
        print("\n\n[ERROR] Training crashed:")
        traceback.print_exc()
        sys.exit(1)

    print("\n")
    # 4. Chat Loop
    chat_loop(model, tokenizer, dev)

if __name__ == "__main__":
    main()
