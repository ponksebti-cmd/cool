"""
train.py — Optimized training loop for Stage 1 baseline.

Optimizations vs. the naive reference loop:
  ┌─────────────────────────────────────────────────────────────────┐
  │  AMP (--amp)         bfloat16 autocast → ~2× throughput        │
  │  Grad accumulation   Simulates N× batch without N× VRAM cost   │
  │  torch.compile       JIT fusion → ~1.5-2× on CUDA/MPS          │
  │  Grad checkpointing  --grad-ckpt → ~30% less VRAM at 30% cost  │
  │  Fused AdamW         torch optimizer with fused=True on CUDA    │
  │  Separate loss log   LM loss | aux loss | z-loss tracked apart  │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    # Fast CPU/MPS test (TINY config, no AMP):
    python train.py

    # 300M on RX 6650 XT / CUDA with all optimizations:
    python train.py --config 300m --device cuda --amp --compile --steps 5000

    # Memory-constrained (8 GB VRAM): gradient checkpointing + accumulation:
    python train.py --config 300m --amp --grad-accum 4 --grad-ckpt --batch-size 2

The script prints per-step:
  - LM loss, MoE aux-loss, Z-loss
  - Gradient norm
  - Tokens per second (across all accumulation micro-steps)
  - Current learning rate
  - (CUDA only) GPU memory allocated
"""

from __future__ import annotations
import argparse
import sys
import time
import math
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, ".")
from src import Transformer, DEFAULT_300M, TINY_TEST, make_dataloader


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimized Stage 1 training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default="tiny", choices=["tiny", "300m"],
                   help="Model size preset")
    p.add_argument("--device",       default="auto",
                   help="Compute device: auto | cpu | cuda | mps")
    p.add_argument("--steps",        type=int,   default=500,
                   help="Number of optimizer steps")
    p.add_argument("--batch-size",   type=int,   default=4,
                   help="Micro-batch size (per accumulation step)")
    p.add_argument("--grad-accum",   type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    p.add_argument("--lr",           type=float, default=3e-4,
                   help="Peak learning rate")
    p.add_argument("--warmup-steps", type=int,   default=50,
                   help="Linear LR warmup steps")
    p.add_argument("--grad-clip",    type=float, default=1.0,
                   help="Gradient norm clipping threshold")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log-every",    type=int,   default=10,
                   help="Print diagnostics every N optimizer steps")

    # ── Efficiency flags ──────────────────────────────────────────────────
    p.add_argument("--amp",       action="store_true",
                   help="Enable bfloat16 Automatic Mixed Precision (recommended on all devices)")
    p.add_argument("--compile",   action="store_true",
                   help="JIT-compile model with torch.compile (best on CUDA; experimental on MPS)")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="Gradient checkpointing: trade ~30%% compute for ~40%% less VRAM")
    return p.parse_args()


def make_synthetic_docs(
    vocab_size: int, block_size: int, n_docs: int = 2000, seed: int = 42
) -> list[list[int]]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    docs = []
    for _ in range(n_docs):
        length = torch.randint(block_size // 8, block_size // 2, (1,), generator=rng).item()
        tokens = torch.randint(2, vocab_size, (int(length),), generator=rng).tolist()
        docs.append(tokens)
    return docs


def grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of all gradients, for diagnostics."""
    total = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return math.sqrt(total)


def collect_moe_losses(model: torch.nn.Module) -> tuple[float, float]:
    """Sum auxiliary and Z-losses across all MoE layers."""
    aux_loss = 0.0
    z_loss   = 0.0
    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "l_aux"):
            v = mlp.l_aux
            aux_loss += v.item() if isinstance(v, torch.Tensor) else v
        if hasattr(mlp, "l_z"):
            v = mlp.l_z
            z_loss += v.item() if isinstance(v, torch.Tensor) else v
    return aux_loss, z_loss


def mem_str(device: torch.device) -> str:
    """Return a compact memory usage string (CUDA only)."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
        reserved  = torch.cuda.memory_reserved(device)  / 1024 ** 3
        return f"mem: {allocated:.2f}/{reserved:.2f} GB"
    if device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1024 ** 3
        return f"mem: {allocated:.2f} GB"
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    # AMP: bfloat16 is preferred over float16 (same dynamic range as fp32,
    # no loss scaling needed). Supported on CUDA Ampere+, MPS (PyTorch ≥ 2.3),
    # and CPU (software emulation — not recommended).
    amp_enabled = args.amp
    amp_dtype   = torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_enabled)
        if amp_enabled
        else nullcontext()
    )

    # ── Config ────────────────────────────────────────────────────────────
    config = TINY_TEST if args.config == "tiny" else DEFAULT_300M

    # Gradient checkpointing: set on config so Transformer.forward respects it
    if args.grad_ckpt:
        config.use_gradient_checkpointing = True

    eff_batch = args.batch_size * args.grad_accum

    print(f"\n{'='*66}")
    print(f"  Stage 1 Training — {args.config.upper()}")
    print(f"{'='*66}")
    print(f"  Device         : {dev}")
    print(f"  Steps          : {args.steps}  (optimizer updates)")
    print(f"  Micro-batch    : {args.batch_size}  ×  {args.grad_accum} accum  =  {eff_batch} effective")
    print(f"  Peak LR        : {args.lr}")
    print(f"  AMP            : {'bfloat16 ✓' if amp_enabled else 'off'}")
    print(f"  torch.compile  : {'✓' if args.compile else 'off'}")
    print(f"  Grad checkpoint: {'✓ (~30% VRAM saving)' if args.grad_ckpt else 'off'}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = Transformer(config).to(dev)
    n_params = model.num_parameters()
    print(f"  Parameters     : {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"{'='*66}\n")

    # Optional JIT compilation — biggest single speedup on CUDA.
    # torch.compile traces the graph, fuses kernels, and eliminates Python
    # overhead in the hot path. Accepts a ~30s warm-up on first step.
    if args.compile:
        if hasattr(torch, "compile"):
            print("  Compiling model with torch.compile (reduce-overhead mode)...")
            print("  First step will be slow (~30s); subsequent steps will be fast.\n")
            model = torch.compile(model, mode="reduce-overhead")
        else:
            print("  WARNING: torch.compile not available (requires PyTorch >= 2.0). Skipping.\n")

    # ── Data ──────────────────────────────────────────────────────────────
    docs = make_synthetic_docs(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_docs=3000,
        seed=args.seed,
    )
    loader = make_dataloader(
        documents=docs,
        block_size=config.block_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    data_iter = iter(loader)

    def next_batch() -> dict:
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Use fused AdamW on CUDA (single-kernel weight update, ~10% faster).
    use_fused = (dev.type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__doc__ or True)
    fused_available = dev.type == "cuda"
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
        fused=fused_available,  # kernel-fused update on CUDA (no-op on MPS/CPU)
    )

    # Cosine decay scheduler (kicks in after warmup)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # MoE loss coefficients (read from config with safe fallbacks)
    moe_aux_coef = getattr(config, "moe_aux_loss_coef", 0.0)
    moe_z_coef   = getattr(config, "moe_z_loss_coef",   0.0)

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    tokens_seen  = 0
    t_start      = time.perf_counter()
    t_last       = t_start

    for step in range(1, args.steps + 1):

        # ── Linear LR warmup ──────────────────────────────────────────────
        if step <= args.warmup_steps:
            lr_scale = step / max(1, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

        # ── Gradient accumulation ─────────────────────────────────────────
        # Zero once per optimizer step, accumulate over micro-steps.
        optimizer.zero_grad(set_to_none=True)

        step_lm_loss  = 0.0
        step_aux_loss = 0.0
        step_z_loss   = 0.0
        step_tokens   = 0

        for micro in range(args.grad_accum):
            batch        = next_batch()
            input_ids    = batch["input_ids"].to(dev, non_blocking=True)
            targets      = batch["targets"].to(dev, non_blocking=True)
            position_ids = batch["position_ids"].to(dev, non_blocking=True)
            attn_mask    = batch["attn_mask"].to(dev, non_blocking=True)

            with autocast_ctx:
                _, lm_loss = model(
                    input_ids,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    targets=targets,
                )
                aux_loss, z_loss = collect_moe_losses(model)
                # Tensors may be returned by MoE layers; detach for logging
                total_loss = (
                    lm_loss
                    + moe_aux_coef * (aux_loss if isinstance(aux_loss, torch.Tensor) else lm_loss.new_tensor(aux_loss))
                    + moe_z_coef   * (z_loss   if isinstance(z_loss,   torch.Tensor) else lm_loss.new_tensor(z_loss))
                )
                # Scale loss for accumulation BEFORE backward so gradients
                # are already averaged (not summed) across micro-steps.
                total_loss = total_loss / args.grad_accum

            total_loss.backward()

            step_lm_loss  += lm_loss.item() / args.grad_accum
            step_aux_loss += (aux_loss if isinstance(aux_loss, float) else aux_loss.item()) / args.grad_accum
            step_z_loss   += (z_loss   if isinstance(z_loss,   float) else z_loss.item())   / args.grad_accum
            step_tokens   += input_ids.numel()

        # ── Optimizer step ────────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        gn = grad_norm(model)
        optimizer.step()

        # Cosine decay (only after warmup)
        if step > args.warmup_steps:
            scheduler.step()

        tokens_seen += step_tokens

        # ── Logging ───────────────────────────────────────────────────────
        if step % args.log_every == 0:
            now          = time.perf_counter()
            elapsed      = now - t_last
            tok_per_sec  = (step_tokens * args.log_every) / elapsed
            t_last       = now
            current_lr   = optimizer.param_groups[0]["lr"]
            mem          = mem_str(dev)

            moe_str = (
                f" | aux: {step_aux_loss:.4f} | zloss: {step_z_loss:.4f}"
                if moe_aux_coef > 0 or moe_z_coef > 0
                else ""
            )

            print(
                f"  step {step:5d}/{args.steps}"
                f" | loss: {step_lm_loss:.4f}"
                f"{moe_str}"
                f" | gnorm: {gn:.3f}"
                f" | lr: {current_lr:.2e}"
                f" | tok/s: {tok_per_sec:>8,.0f}"
                + (f" | {mem}" if mem else "")
            )

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    print(f"\n{'─'*66}")
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Total tokens processed : {tokens_seen:,}")
    print(f"  Average tok/s          : {tokens_seen / total_time:,.0f}")
    print(f"{'─'*66}\n")


if __name__ == "__main__":
    main()
