#!/usr/bin/env python3
"""
train_real.py — Production-grade pre-training loop.

Features:
  - Streaming dataset: HuggingFaceFW/fineweb-edu (no massive local downloads).
  - Tokenization: Mistral-7B tokenizer (32000 vocab size).
  - Optimized for 8GB VRAM (AMD RX 6650 XT or similar):
      - Gradient checkpointing default (saves ~40% VRAM).
      - Gradient accumulation (simulates large batches).
      - bfloat16 AMP (maximises memory bandwidth).
  - Auto-checkpoints and graceful Ctrl+C (KeyboardInterrupt) saving.

Usage:
    python3 train_real.py --config 300m --batch-size 2 --grad-accum 16
"""

from __future__ import annotations
import argparse
import sys
import time
import math
import os
import signal
from contextlib import nullcontext
from typing import Iterator

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
from src import Transformer, DEFAULT_300M, MAC_NANO, TINY_TEST


# ── Configuration ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config",       default="300m", choices=["tiny", "nano", "300m"])
    p.add_argument("--device",       default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--steps",        type=int,   default=100_000, help="Total training steps")
    p.add_argument("--batch-size",   type=int,   default=4,  help="Micro-batch size (T4 16GB can handle 4-8)")
    p.add_argument("--grad-accum",   type=int,   default=8,  help="Gradient accumulation steps")
    p.add_argument("--lr",           type=float, default=4e-4, help="Peak learning rate")
    p.add_argument("--warmup-steps", type=int,   default=1000, help="Linear LR warmup steps")
    p.add_argument("--grad-clip",    type=float, default=1.0, help="Gradient clipping norm")
    p.add_argument("--save-every",   type=int,   default=2000, help="Steps between auto-saves")
    p.add_argument("--log-every",    type=int,   default=10,  help="Steps between terminal logs")
    p.add_argument("--dtype",        default="fp16", choices=["fp16", "bf16"],
                   help="fp16 = Tensor Core optimized for T4/V100; bf16 = better for A100/H100")
    p.add_argument("--no-amp",       action="store_true", help="Disable mixed precision (not recommended)")
    p.add_argument("--no-ckpt",      action="store_true", help="Disable gradient checkpointing (uses MORE VRAM)")
    p.add_argument("--compile",      action="store_true", help="Enable torch.compile (requires PyTorch 2.0+)")
    return p.parse_args()


# ── Data Pipeline (Streaming + Packing) ───────────────────────────────────────

def stream_packed_batches(
    tokenizer: AutoTokenizer,
    batch_size: int,
    block_size: int,
    dataset_name: str = "JeanKaddour/minipile",
    config_name: str = "default",
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Streams data from HF, tokenizes on the fly, and packs tokens densely
    into batches of shape [batch_size, block_size].
    
    This is continuous packing (documents cross block boundaries). We do
    not use block-diagonal masking here for simplicity and raw throughput.
    """
    # Load dataset in streaming mode (no local disk download)
    dataset = load_dataset(dataset_name, name=config_name, split="train", streaming=True)
    
    buffer = []
    
    for row in dataset:
        text = row["text"]
        # Tokenize (returns a list of integers)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        buffer.extend(tokens)
        
        # When we have enough tokens to form a full batch
        tokens_needed = batch_size * block_size
        while len(buffer) >= tokens_needed:
            # Slice out the tokens
            chunk = buffer[:tokens_needed]
            buffer = buffer[tokens_needed:]
            
            # Reshape into [batch_size, block_size]
            input_ids = torch.tensor(chunk, dtype=torch.long).view(batch_size, block_size)
            
            # For continuous packing, targets are just input_ids shifted by 1.
            # (Note: the very last target of each block is technically the first
            # token of the next block, but shifting within the block is standard
            # for continuous packing without cross-batch state).
            targets = torch.empty_like(input_ids)
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = tokenizer.eos_token_id or 0
            
            # Absolute positions
            position_ids = torch.arange(block_size, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            
            yield {
                "input_ids": input_ids,
                "targets": targets,
                "position_ids": position_ids,
                "attn_mask": None, # Standard causal mask will be used internally
            }


# ── Helpers ───────────────────────────────────────────────────────────────────

def grad_norm(model: torch.nn.Module) -> float:
    total = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
    return math.sqrt(total)


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


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int):
    print(f"\n[SAVE] Saving checkpoint to {path} ...")
    # Handle compiled models
    state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }, path)
    print(f"[SAVE] Checkpoint saved successfully.")


# ── Graceful Shutdown via Signal ──────────────────────────────────────────────
# We use a flag-based approach instead of relying on KeyboardInterrupt inside
# a try/except block, because MPS/CUDA GPU kernels block Python's signal
# delivery until the entire C++ kernel finishes. By using signal.signal()
# we can set a flag the moment Ctrl+C is pressed, then check it between
# micro-batches so the script exits cleanly within ~30 seconds instead of
# waiting for the full 2-minute step.
_shutdown_requested = False

def _sigint_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        # Second Ctrl+C — force quit immediately
        print("\n\n[FATAL] Second Ctrl+C received. Force-quitting NOW (checkpoint may be lost).")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n[SAVE] Ctrl+C caught — will save checkpoint after current micro-batch finishes...")
    print("        (Press Ctrl+C again to force-quit immediately, risking checkpoint loss.)")


def print_progress(step: int, total: int, loss: float, t0: float, tokens: int, lr: float):
    elapsed = time.perf_counter() - t0
    tok_per_sec = tokens / elapsed if elapsed > 0 else 0
    percent = step / total
    
    if step > 0:
        total_time_est = elapsed / percent
        eta_seconds = total_time_est - elapsed
        m, s = divmod(int(eta_seconds), 60)
        h, m = divmod(m, 60)
        eta_str = f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"
    else:
        eta_str = "..."

    bar_len = 25
    filled_len = int(bar_len * percent)
    bar = '█' * filled_len + '░' * (bar_len - filled_len)
    
    # Use carriage return \r to overwrite the line
    sys.stdout.write(
        f"\r  [{bar}] {percent*100:3.0f}% | "
        f"Loss: {loss:.4f} | "
        f"Tok/s: {tok_per_sec:,.0f} | "
        f"LR: {lr:.2e} | "
        f"ETA: {eta_str}    "
    )
    sys.stdout.flush()

# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    # Device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dev = torch.device(device)

    # Config
    if args.config == "tiny":
        config = TINY_TEST
    elif args.config == "nano":
        config = MAC_NANO
    else:
        config = DEFAULT_300M
    if not args.no_ckpt:
        config.use_gradient_checkpointing = True

    # ── NVIDIA T4 / CUDA Hardware Optimizations ────────────────────────────
    if dev.type == "cuda":
        # cuDNN benchmark mode: on first run it tries several convolution/matmul
        # algorithms and picks the fastest for your exact tensor shapes.
        torch.backends.cudnn.benchmark = True
        # TF32 is NOT supported on T4 (Turing). It is Ampere+ only.
        # Explicitly disable so PyTorch doesn't silently downgrade precision.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Model
    print("Loading model...")
    model = Transformer(config).to(dev)
    
    # torch.compile: tracing the graph lets the compiler fuse ops and eliminate
    # Python overhead. "max-autotune" benchmarks multiple kernel implementations
    # at startup (~2-3 min) then runs the absolute fastest version afterwards.
    if args.compile and hasattr(torch, "compile"):
        print("[COMPILE] Running torch.compile(mode='max-autotune') — may take 2-3 min at startup...")
        model = torch.compile(model, mode="max-autotune")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # fused=True on CUDA: runs the optimizer kernel in a single GPU pass instead
    # of one pass per parameter tensor. Can be 2-5x faster for large models.
    fused_available = (dev.type == "cuda")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-8, fused=fused_available
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # Tokenizer & Data
    print("Loading tokenizer (mistralai/Mistral-7B-v0.1)...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    print("Connecting to streaming dataset (HuggingFaceFW/fineweb-edu)...")
    batch_iterator = stream_packed_batches(tokenizer, args.batch_size, config.block_size)

    # ── AMP (Mixed Precision) ─────────────────────────────────────────────────
    # CRITICAL T4 note: The T4 is Turing architecture.
    # - FP16 uses dedicated Tensor Cores → ~8x speedup over FP32
    # - BF16 is NOT hardware-accelerated on T4 (that's Ampere/A100+)
    # - FP16 requires an explicit GradScaler to prevent underflow
    use_amp = (dev.type in ("cuda", "mps")) and not args.no_amp
    amp_dtype = torch.float16 if (args.dtype == "fp16" and dev.type == "cuda") else torch.bfloat16
    amp_ctx = torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()
    # GradScaler prevents FP16 gradient underflow. Only used for CUDA FP16.
    use_scaler = (use_amp and amp_dtype == torch.float16 and dev.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    amp_label = ("fp16 ✦ Tensor Cores" if amp_dtype == torch.float16 else "bfloat16") if use_amp else "off"
    eff_batch = args.batch_size * args.grad_accum
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*62}")
    print(f"  PRODUCTION PRE-TRAINING  ──  NVIDIA T4 OPTIMIZED")
    print(f"{'='*62}")
    print(f"  Device      : {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''})")  
    print(f"  Parameters  : {param_count/1e6:.1f}M")
    print(f"  Micro-batch : {args.batch_size}  x  {args.grad_accum} accum = {eff_batch} effective")
    print(f"  AMP         : {amp_label}")
    print(f"  GradScaler  : {'✓' if use_scaler else 'off'}")
    print(f"  Fused Adam  : {'✓' if fused_available else 'off'}")
    print(f"  Grad Ckpt   : {'✓' if config.use_gradient_checkpointing else 'off'}")
    print(f"  Compiled    : {'✓ max-autotune' if args.compile else 'off (add --compile for ~20% speedup)'}")
    print(f"{'='*62}\n")

    model.train()
    tokens_seen = 0
    t_start = time.perf_counter()
    t_last = t_start
    
    # Register the graceful shutdown signal handler
    signal.signal(signal.SIGINT, _sigint_handler)
    
    moe_aux_coef = getattr(config, "moe_aux_loss_coef", 0.0)
    moe_z_coef   = getattr(config, "moe_z_loss_coef",   0.0)

    step = 0
    
    print("\n[INFO] Initializing Fast-Stream Dataset (MiniPile)...")
    print("[INFO] This dataset uses small shards and will start almost instantly.\n")
    
    try:
        for step in range(1, args.steps + 1):
            
            # Warmup
            if step <= args.warmup_steps:
                lr_scale = step / max(1, args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            optimizer.zero_grad(set_to_none=True)
            step_lm_loss, step_tokens = 0.0, 0

            # Gradient Accumulation
            for i in range(args.grad_accum):
                # Check shutdown flag BETWEEN micro-batches (not mid-kernel)
                if _shutdown_requested:
                    break
                    
                if step == 1:
                    # Print micro-batch progress for the very first step so the user
                    # knows the GPU is calculating and the script is not frozen.
                    sys.stdout.write(f"\r  [INFO] Crunching Micro-batch {i+1}/{args.grad_accum} on M2 GPU...")
                    sys.stdout.flush()
                    
                batch = next(batch_iterator)
                input_ids    = batch["input_ids"].to(dev, non_blocking=True)
                targets      = batch["targets"].to(dev, non_blocking=True)
                position_ids = batch["position_ids"].to(dev, non_blocking=True)
                
                with amp_ctx:
                    _, lm_loss = model(input_ids, position_ids=position_ids, targets=targets)
                    aux_loss, z_loss = collect_moe_losses(model)
                    total_loss = lm_loss + (moe_aux_coef * aux_loss) + (moe_z_coef * z_loss)
                    total_loss = total_loss / args.grad_accum
                
                # GradScaler scales the loss to prevent FP16 underflow,
                # then unscales before the optimizer step.
                scaler.scale(total_loss).backward()
                step_lm_loss += lm_loss.item() / args.grad_accum
                step_tokens += input_ids.numel()

            # Unscale before clipping so grad norms are in true FP32 scale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            gn = grad_norm(model)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if step > args.warmup_steps:
                scheduler.step()

            tokens_seen += step_tokens

            # Check if user requested shutdown after completing an optimizer step
            if _shutdown_requested:
                print("\n")
                save_checkpoint("ckpt_interrupted.pt", model, optimizer, step)
                print("[INFO] Saved! You can resume training later.")
                sys.exit(0)

            # Logging (always log step 1 so the user gets immediate feedback)
            if step == 1 or step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print_progress(step, args.steps, step_lm_loss, t_start, tokens_seen, current_lr)

            # Auto-save
            if step % args.save_every == 0:
                save_checkpoint(f"ckpt_step_{step}.pt", model, optimizer, step)
                
    except KeyboardInterrupt:
        # Fallback in case signal handler wasn't registered or second Ctrl+C
        print("\n\n[WARNING] Interrupted. Saving checkpoint...")
        save_checkpoint("ckpt_interrupted.pt", model, optimizer, step)
        print("Exiting cleanly.")
        sys.exit(0)

    print("\nTraining Complete.")
    save_checkpoint("ckpt_final.pt", model, optimizer, step)


if __name__ == "__main__":
    main()
