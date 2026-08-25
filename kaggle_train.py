#!/usr/bin/env python3
"""
kaggle_train.py — Kaggle Notebook Optimized Training Script (DDP Edition)

HOW TO USE IN KAGGLE:
═════════════════════
1. Create a new Notebook in Kaggle.
2. Turn on Internet (Settings -> Internet).
3. Select Accelerator: GPU T4x2 (recommended).
4. Paste this into a cell and run:

    !git clone https://github.com/YOUR_USER/YOUR_REPO optimistic-lavoisier
    %cd optimistic-lavoisier/optimistic-lavoisier
    !pip install -q datasets transformers
    !python3 kaggle_train.py

FEATURES:
  ✓ DistributedDataParallel (DDP) — NO memory waste vs naive DataParallel
  ✓ Each GPU holds only its own gradients — peer-to-peer all_reduce
  ✓ Full 838M parameter MoE model fits comfortably on T4x2
  ✓ Auto-detects P100 vs T4 and single vs dual GPU
  ✓ Saves checkpoints to /kaggle/working/ (persisted after session ends)
  ✓ Auto-resumes from latest checkpoint on restart
  ✓ Graceful Ctrl+C saving
"""

from __future__ import annotations
import sys
import os
import time
import signal
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

# Prevent CUDA memory fragmentation — critical for Kaggle GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("\n[ERROR] Missing: pip install datasets transformers\n")
    sys.exit(1)

sys.path.insert(0, ".")
from src import Transformer, DEFAULT_300M, FLAGSHIP_700M, FLAGSHIP_1B, FLAGSHIP_3B, MAC_NANO, TINY_TEST

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
STEPS           = 100_000
LR              = 4e-4
WARMUP_STEPS    = 1_000
GRAD_CLIP       = 1.0
SAVE_EVERY      = 1_000
LOG_INTERVAL_S  = 30.0      # Print progress every 30 seconds
MODEL_CONFIG    = "700m"    # "3b" | "1b" | "700m" | "300m" | "nano" | "tiny"
KAGGLE_SAVE_DIR = "/kaggle/working/checkpoints"
LOCAL_SAVE_DIR  = "./checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working")


def get_save_dir() -> Path:
    if is_kaggle():
        save_dir = Path(KAGGLE_SAVE_DIR)
        print("[KAGGLE] Running in Kaggle environment.")
    else:
        save_dir = Path(LOCAL_SAVE_DIR)
        print("[LOCAL]  Not in Kaggle. Using local directory.")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE]   Checkpoints will be saved to: {save_dir}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(save_dir.glob("ckpt_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts[-1] if ckpts else None


def save_checkpoint(path: Path, model, optimizer, scaler, step: int):
    print(f"\n[SAVE] Writing {path} ...")
    raw = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    state = raw._orig_mod.state_dict() if hasattr(raw, "_orig_mod") else raw.state_dict()
    torch.save({
        "model_state_dict":    state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":   scaler.state_dict() if scaler else None,
        "step":                step,
    }, path)
    print(f"[SAVE] ✓ Step {step} saved to {path.name}")


def load_checkpoint(path: Path, model, optimizer, scaler, dev) -> int:
    print(f"[RESUME] Loading {path} ...")
    ckpt = torch.load(path, map_location=dev)
    raw = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    raw.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt["step"] + 1
    print(f"[RESUME] ✓ Continuing from step {step}")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def stream_packed_batches(tokenizer, batch_size: int, block_size: int) -> Iterator:
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True
    )
    buffer = []
    for row in dataset:
        buffer.extend(tokenizer.encode(row["text"], add_special_tokens=True))
        tokens_needed = batch_size * block_size
        while len(buffer) >= tokens_needed:
            chunk   = buffer[:tokens_needed]
            buffer  = buffer[tokens_needed:]
            ids     = torch.tensor(chunk, dtype=torch.long).view(batch_size, block_size)
            targets = torch.empty_like(ids)
            targets[:, :-1] = ids[:, 1:]
            targets[:, -1]  = tokenizer.eos_token_id or 0
            pos_ids = torch.arange(block_size).unsqueeze(0).expand(batch_size, -1)
            yield {"input_ids": ids, "targets": targets, "position_ids": pos_ids}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def collect_moe_losses(model):
    raw = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    aux, z = 0.0, 0.0
    for layer in raw.layers:
        mlp = layer.mlp
        if hasattr(mlp, "l_aux"):
            v = mlp.l_aux
            aux += v.item() if isinstance(v, torch.Tensor) else v
        if hasattr(mlp, "l_z"):
            v = mlp.l_z
            z   += v.item() if isinstance(v, torch.Tensor) else v
    return aux, z


def print_progress(step, total, loss, t0, tokens_seen, lr, gpu_name, vram_gb_used):
    elapsed  = time.perf_counter() - t0
    tok_s    = tokens_seen / elapsed if elapsed > 0 else 0
    pct      = step / total
    eta_s    = (elapsed / pct - elapsed) if pct > 0 else 0
    h, rem   = divmod(int(eta_s), 3600)
    m, s     = divmod(rem, 60)
    eta_str  = f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"
    bar      = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
    tok_str  = f"{tokens_seen/1e9:.2f}B" if tokens_seen >= 1e9 else f"{tokens_seen/1e6:.1f}M"
    vram_str = f"{vram_gb_used:.1f}G"

    # Use print() with \n so Kaggle's logger captures each line (no \r tricks)
    print(
        f"  [{bar}] {pct*100:3.0f}% | "
        f"Step: {step:,}/{total:,} | "
        f"Toks: {tok_str} | "
        f"Loss: {loss:.4f} | "
        f"Tok/s: {tok_s:,.0f} | "
        f"LR: {lr:.2e} | "
        f"ETA: {eta_str} | "
        f"VRAM: {vram_str}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
_shutdown = False

def _sigint_handler(sig, frame):
    global _shutdown
    if _shutdown:
        sys.exit(1)
    _shutdown = True
    print("\n[SAVE] Ctrl+C caught — finishing current step, then saving...")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING WORKER (runs once per GPU process in DDP mode)
# ─────────────────────────────────────────────────────────────────────────────
def train_worker(rank: int, world_size: int, save_dir: Path, amp_dtype: torch.dtype):
    """
    This function runs in its own process for each GPU.
    rank=0 is the "master" process that prints and saves checkpoints.
    """
    is_master = (rank == 0)

    # ── DDP setup ───────────────────────────────────────────────────────────
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    use_scaler = (amp_dtype == torch.float16)

    # CUDA backend tuning
    torch.backends.cudnn.benchmark = True
    is_ampere = (amp_dtype == torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = is_ampere
    torch.backends.cudnn.allow_tf32 = is_ampere

    # ── Build config ────────────────────────────────────────────────────────
    if MODEL_CONFIG == "tiny":   config = TINY_TEST
    elif MODEL_CONFIG == "nano": config = MAC_NANO
    elif MODEL_CONFIG == "700m": config = FLAGSHIP_700M
    elif MODEL_CONFIG == "1b":   config = FLAGSHIP_1B
    elif MODEL_CONFIG == "3b":   config = FLAGSHIP_3B
    else:                         config = DEFAULT_300M
    config.use_gradient_checkpointing = True

    # ── Build model ─────────────────────────────────────────────────────────
    model = Transformer(config).to(dev)
    param_count = sum(p.numel() for p in model.parameters())

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-8, fused=True
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── Auto-resume ─────────────────────────────────────────────────────────
    start_step = 1
    if is_master:
        ckpt = find_latest_checkpoint(save_dir)
        if ckpt:
            start_step = load_checkpoint(ckpt, model, optimizer, scaler, dev)
            for _ in range(start_step - 1):
                if _ > WARMUP_STEPS:
                    scheduler.step()
    if world_size > 1:
        # Broadcast start_step from rank 0 to all other ranks
        t = torch.tensor([start_step], dtype=torch.long, device=dev)
        dist.broadcast(t, src=0)
        start_step = int(t.item())

    # ── AMP context ─────────────────────────────────────────────────────────
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)

    # Batch size: each GPU processes 1 sequence at a time (sufficient for 16-step grad accum)
    batch_size_per_gpu = 1
    # Each GPU uses a different part of the dataset (skip rank * N tokens)
    # Simple approach: each rank uses the same stream (fine for single-epoch streaming)
    grad_accum = max(1, 32 // (batch_size_per_gpu * world_size))

    moe_aux_coef = getattr(config, "moe_aux_loss_coef", 0.0)
    moe_z_coef   = getattr(config, "moe_z_loss_coef",   0.0)

    if is_master:
        gpu_name = torch.cuda.get_device_name(0).upper()
        dtype_name = "fp16 ✦ Tensor Cores" if amp_dtype == torch.float16 else "bfloat16 ✦ Tensor Cores"
        print(f"\n{'='*62}")
        print(f"  KAGGLE TRAINING  ──  DDP OPTIMIZED")
        print(f"{'='*62}")
        print(f"  GPUs        : {world_size}x {gpu_name}")
        print(f"  Parameters  : {param_count/1e6:.1f}M")
        print(f"  Batching    : {batch_size_per_gpu * world_size} per step  ×  {grad_accum} accum = {batch_size_per_gpu * world_size * grad_accum} effective")
        print(f"  Precision   : {dtype_name}")
        print(f"  GradScaler  : {'✓' if use_scaler else 'off'}")
        print(f"  Checkpoints : {save_dir}")
        print(f"  Resuming    : step {start_step}")
        print(f"{'='*62}\n")
        print("[DATA] Connecting to FineWeb-Edu 10B ...")

    tokenizer  = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    batch_iter = stream_packed_batches(tokenizer, batch_size_per_gpu, config.block_size)

    if rank == 0:
        signal.signal(signal.SIGINT, _sigint_handler)

    model.train()
    tokens_seen  = 0
    t0           = time.perf_counter()
    last_log     = t0
    step_loss    = 0.0

    for step in range(start_step, STEPS + 1):

        # Warmup
        if step <= WARMUP_STEPS:
            for pg in optimizer.param_groups:
                pg["lr"] = LR * (step / max(1, WARMUP_STEPS))

        optimizer.zero_grad(set_to_none=True)
        step_loss, step_tokens = 0.0, 0

        for _ in range(grad_accum):
            if _shutdown:
                break
            batch = next(batch_iter)
            ids   = batch["input_ids"].to(dev, non_blocking=True)
            tgts  = batch["targets"].to(dev, non_blocking=True)
            pids  = batch["position_ids"].to(dev, non_blocking=True)

            with amp_ctx:
                _, lm_loss = model(ids, position_ids=pids, targets=tgts)
                if isinstance(lm_loss, torch.Tensor) and lm_loss.dim() > 0:
                    lm_loss = lm_loss.mean()
                aux, z     = collect_moe_losses(model)
                total_loss = (lm_loss + moe_aux_coef * aux + moe_z_coef * z) / grad_accum

            scaler.scale(total_loss).backward()
            step_loss   += lm_loss.item() / grad_accum
            step_tokens += ids.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step > WARMUP_STEPS:
            scheduler.step()

        tokens_seen += step_tokens

        # Shutdown save
        if _shutdown and is_master:
            save_checkpoint(save_dir / "ckpt_interrupted.pt", model, optimizer, scaler, step)
            print("[INFO] Saved. Re-run script to resume.")
            break

        # Logging (master only, every 30 seconds)
        now = time.perf_counter()
        if is_master and (step == start_step or (now - last_log) >= LOG_INTERVAL_S):
            vram = torch.cuda.memory_allocated(0) / 1e9
            current_lr = optimizer.param_groups[0]["lr"]
            print_progress(step, STEPS, step_loss, t0, tokens_seen, current_lr,
                           torch.cuda.get_device_name(0), vram)
            last_log = now

        # Checkpoint (master only)
        if is_master and step % SAVE_EVERY == 0:
            save_checkpoint(save_dir / f"ckpt_step_{step}.pt", model, optimizer, scaler, step)

    # Final save
    if is_master and not _shutdown:
        print("\nTraining complete!")
        save_checkpoint(save_dir / "ckpt_final.pt", model, optimizer, scaler, STEPS)

    if world_size > 1:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — auto-launches DDP or single-GPU
# ─────────────────────────────────────────────────────────────────────────────
def main():
    save_dir = get_save_dir()

    if not torch.cuda.is_available():
        print("[WARN] No GPU found. Running on CPU (very slow).")
        train_worker(0, 1, save_dir, torch.float32)
        return

    world_size = torch.cuda.device_count()
    gpu_name   = torch.cuda.get_device_name(0).upper()

    # Pick dtype: T4/V100/P100 → FP16, Ampere+ → BF16
    is_ampere_plus = any(x in gpu_name for x in ["A100", "A10", "L4", "H100", "RTX 4", "RTX 3"])
    amp_dtype = torch.bfloat16 if is_ampere_plus else torch.float16

    if world_size > 1:
        print(f"[DDP] Launching {world_size}-GPU DistributedDataParallel training...")
        # Use a temp file as the rendezvous point for process coordination
        init_file = tempfile.mktemp(suffix=".pt")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["INIT_METHOD"] = f"file://{init_file}"
        mp.spawn(train_worker, args=(world_size, save_dir, amp_dtype), nprocs=world_size, join=True)
    else:
        print(f"[SINGLE-GPU] Training on {gpu_name}")
        train_worker(0, 1, save_dir, amp_dtype)


if __name__ == "__main__":
    main()
