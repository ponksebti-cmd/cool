#!/usr/bin/env python3
"""
kaggle_train.py — Kaggle Notebook Optimized Training Script

HOW TO USE IN KAGGLE:
═════════════════════
1. Create a new Notebook in Kaggle.
2. Turn on the Internet in the right-hand panel (Settings -> Internet).
3. Select your Accelerator (Settings -> Accelerator).
   - T4x2 is highly recommended (Tensor Cores + 32GB total VRAM).
   - P100 is older and slower (No Tensor Cores).
4. Paste this code into a cell and run it:

    !git clone https://github.com/YOUR_USER/YOUR_REPO optimistic-lavoisier
    %cd optimistic-lavoisier
    !pip install -q datasets transformers
    !python3 kaggle_train.py

FEATURES:
  ✓ Auto-detects Kaggle environment and saves to /kaggle/working/ (which persists).
  ✓ Auto-detects P100 vs T4 GPUs.
  ✓ P100: Falls back to FP32 automatically (since P100 lacks Tensor Cores).
  ✓ T4x2: Automatically utilizes BOTH T4 GPUs via DataParallel for double speed!
  ✓ Auto-resumes from latest checkpoint in /kaggle/working/ if restarted.
"""

from __future__ import annotations
import sys
import os
import time
import math
import signal
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

# Prevent CUDA memory fragmentation on Kaggle GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
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
# CONFIG — adjust these to your needs
# ─────────────────────────────────────────────────────────────────────────────
STEPS          = 100_000
LR             = 4e-4
WARMUP_STEPS   = 1_000
GRAD_CLIP      = 1.0
SAVE_EVERY     = 1000       # save every 1000 steps
MODEL_CONFIG   = "700m"     # "3b" | "1b" | "700m" | "300m" | "nano" | "tiny"
KAGGLE_SAVE_DIR = "/kaggle/working/checkpoints"
LOCAL_SAVE_DIR  = "./checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE DETECTION & SAVE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working")


def get_save_dir() -> Path:
    """Returns the persistent save directory."""
    if is_kaggle():
        save_dir = Path(KAGGLE_SAVE_DIR)
        print("[KAGGLE] Running in Kaggle environment.")
    else:
        save_dir = Path(LOCAL_SAVE_DIR)
        print("[LOCAL] Not in Kaggle. Using local directory.")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE] Checkpoints will be saved to: {save_dir}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# GPU AUTO-DETECTION & MULTI-GPU (T4x2)
# ─────────────────────────────────────────────────────────────────────────────
def detect_gpu_config() -> dict:
    """
    Kaggle typically offers P100 (16GB) or T4x2 (2x 16GB).
    - P100 (Pascal): NO Tensor Cores. FP16 is slow/emulated. Use FP32.
    - T4 (Turing): Has FP16 Tensor Cores.
    """
    if not torch.cuda.is_available():
        return {"device": "cpu", "dtype": torch.float32, "batch_size": 1,
                "grad_accum": 4, "use_scaler": False, "name": "CPU", "num_gpus": 0}
    
    num_gpus = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0).upper()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    is_pascal = "P100" in name
    is_t4 = "T4" in name
    
    if is_pascal:
        dtype = torch.float32
        use_scaler = False
        batch_size = 1
    elif is_t4:
        dtype = torch.float16
        use_scaler = True
        batch_size = 1
    else:
        # Fallback for newer GPUs (A100 etc if Kaggle adds them)
        dtype = torch.bfloat16
        use_scaler = False
        batch_size = 4
        
    # Scale batch size if multiple GPUs are available (DataParallel)
    effective_batch = batch_size * num_gpus if num_gpus > 0 else batch_size
    grad_accum = max(1, 32 // effective_batch)

    return {
        "device": "cuda",
        "dtype": dtype,
        "batch_size": batch_size, # Per GPU
        "grad_accum": grad_accum,
        "use_scaler": use_scaler,
        "name": name,
        "num_gpus": num_gpus,
        "vram_gb": vram_gb * num_gpus, # Total across all GPUs
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(save_dir: Path) -> Path | None:
    checkpoints = sorted(save_dir.glob("ckpt_step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(path: Path, model, optimizer, scaler, step: int):
    sys.stdout.write("\n")
    print(f"[SAVE] Saving to {path} ...")
    # Unwrap DataParallel if used
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    state_dict = raw_model._orig_mod.state_dict() if hasattr(raw_model, "_orig_mod") else raw_model.state_dict()
    
    torch.save({
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "step": step,
    }, path)
    print(f"[SAVE] ✓ Saved! Step {step}")


def load_checkpoint(path: Path, model, optimizer, scaler, dev) -> int:
    print(f"[RESUME] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=dev)
    
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        
    start_step = ckpt["step"] + 1
    print(f"[RESUME] ✓ Resuming from step {start_step}")
    return start_step


# ─────────────────────────────────────────────────────────────────────────────
# DATA STREAMING
# ─────────────────────────────────────────────────────────────────────────────
def stream_packed_batches(tokenizer, batch_size: int, block_size: int, num_gpus: int) -> Iterator:
    # If using DataParallel, we must provide larger batches so they can be split across GPUs
    actual_batch = batch_size * max(1, num_gpus)
    
    # Using FineWeb-Edu (Sample-10BT) for extremely high quality educational text.
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buffer = []
    for row in dataset:
        tokens = tokenizer.encode(row["text"], add_special_tokens=True)
        buffer.extend(tokens)
        tokens_needed = actual_batch * block_size
        
        while len(buffer) >= tokens_needed:
            chunk = buffer[:tokens_needed]
            buffer = buffer[tokens_needed:]
            input_ids = torch.tensor(chunk, dtype=torch.long).view(actual_batch, block_size)
            targets = torch.empty_like(input_ids)
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = tokenizer.eos_token_id or 0
            pos_ids = torch.arange(block_size).unsqueeze(0).expand(actual_batch, -1)
            yield {"input_ids": input_ids, "targets": targets, "position_ids": pos_ids}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def collect_moe_losses(model):
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    aux_loss, z_loss = 0.0, 0.0
    for layer in raw_model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "l_aux"):
            v = mlp.l_aux
            aux_loss += v.item() if isinstance(v, torch.Tensor) else v
        if hasattr(mlp, "l_z"):
            v = mlp.l_z
            z_loss += v.item() if isinstance(v, torch.Tensor) else v
    return aux_loss, z_loss


def print_progress(step, total, loss, t0, tokens_seen, lr, gpu_cfg):
    elapsed = time.perf_counter() - t0
    tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
    percent = step / total
    eta_s = (elapsed / percent - elapsed) if percent > 0 else 0
    h, rem = divmod(int(eta_s), 3600)
    m, s = divmod(rem, 60)
    eta_str = f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"
    bar = "█" * int(20 * percent) + "░" * (20 - int(20 * percent))
    
    # Format tokens seen (M or B)
    if tokens_seen >= 1e9:
        tok_str = f"{tokens_seen/1e9:.2f}B"
    else:
        tok_str = f"{tokens_seen/1e6:.1f}M"
        
    vram = f" | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}G" if torch.cuda.is_available() else ""
    
    sys.stdout.write(
        f"\r  [{bar}] {percent*100:3.0f}% | Step: {step} | Toks: {tok_str} | "
        f"Loss: {loss:.4f} | Tok/s: {tok_per_sec:,.0f} | LR: {lr:.2e} | ETA: {eta_str}{vram}   "
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
_shutdown_requested = False

def _sigint_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[FATAL] Force quit.")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n[SAVE] Ctrl+C caught — saving after current micro-batch...")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    save_dir = get_save_dir()
    gpu_cfg = detect_gpu_config()
    
    dev = torch.device(gpu_cfg["device"])
    amp_dtype = gpu_cfg["dtype"]
    base_batch_size = gpu_cfg["batch_size"]
    grad_accum = gpu_cfg["grad_accum"]
    use_scaler = gpu_cfg["use_scaler"]
    num_gpus = gpu_cfg["num_gpus"]

    # CUDA backend tuning
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        is_ampere = amp_dtype == torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = is_ampere
        torch.backends.cudnn.allow_tf32 = is_ampere

    # Load model config
    if MODEL_CONFIG == "tiny":
        config = TINY_TEST
    elif MODEL_CONFIG == "nano":
        config = MAC_NANO
    elif MODEL_CONFIG == "700m":
        config = FLAGSHIP_700M
    elif MODEL_CONFIG == "1b":
        config = FLAGSHIP_1B
    elif MODEL_CONFIG == "3b":
        config = FLAGSHIP_3B
    else:
        config = DEFAULT_300M
        
    config.use_gradient_checkpointing = True

    # Build model
    print("\nBuilding model...")
    model = Transformer(config).to(dev)
    param_count = sum(p.numel() for p in model.parameters())
    
    # Wrap in DataParallel if multiple GPUs (e.g. Kaggle T4x2)
    if num_gpus > 1:
        print(f"[MULTI-GPU] Detected {num_gpus} GPUs. Wrapping model in DataParallel.")
        model = nn.DataParallel(model)

    # Optimizer (fused on CUDA)
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-8,
        fused=(dev.type == "cuda")
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if use_scaler else None

    # Auto-resume
    start_step = 1
    latest_ckpt = find_latest_checkpoint(save_dir)
    if latest_ckpt:
        start_step = load_checkpoint(latest_ckpt, model, optimizer, scaler, dev)
        for _ in range(start_step - 1):
            if _ > WARMUP_STEPS:
                scheduler.step()
    
    # AMP context (disabled on P100)
    use_amp = (amp_dtype != torch.float32) and (dev.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()

    # Reporting
    dtype_name = "fp32 (Pascal fallback)"
    if amp_dtype == torch.float16: dtype_name = "fp16 ✦ Tensor Cores"
    if amp_dtype == torch.bfloat16: dtype_name = "bfloat16 ✦ Tensor Cores"
    
    actual_batch = base_batch_size * max(1, num_gpus)
    
    print(f"\n{'='*62}")
    print(f"  KAGGLE TRAINING  ──  HARDWARE OPTIMIZED")
    print(f"{'='*62}")
    print(f"  GPUs        : {num_gpus}x {gpu_cfg.get('name', 'N/A')} ({gpu_cfg.get('vram_gb', 0):.0f} GB total)")
    print(f"  Parameters  : {param_count/1e6:.1f}M")
    print(f"  Batching    : {actual_batch} per step  ×  {grad_accum} accum = {actual_batch*grad_accum} effective")
    print(f"  Precision   : {dtype_name}")
    print(f"  GradScaler  : {'✓' if use_scaler else 'off'}")
    print(f"  Checkpoints : {save_dir}")
    print(f"{'='*62}\n")

    print("[DATA] Connecting to dataset (FineWeb-Edu 10B)...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    batch_iter = stream_packed_batches(tokenizer, base_batch_size, config.block_size, num_gpus)

    signal.signal(signal.SIGINT, _sigint_handler)
    
    moe_aux_coef = getattr(config, "moe_aux_loss_coef", 0.0)
    moe_z_coef   = getattr(config, "moe_z_loss_coef",   0.0)

    model.train()
    tokens_seen = 0
    t0 = time.perf_counter()
    last_log_time = t0

    try:
        for step in range(start_step, STEPS + 1):

            if step <= WARMUP_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step / max(1, WARMUP_STEPS))

            optimizer.zero_grad(set_to_none=True)
            step_loss, step_tokens = 0.0, 0

            for i in range(grad_accum):
                if _shutdown_requested:
                    break
                batch = next(batch_iter)
                input_ids    = batch["input_ids"].to(dev, non_blocking=True)
                targets      = batch["targets"].to(dev, non_blocking=True)
                position_ids = batch["position_ids"].to(dev, non_blocking=True)

                with amp_ctx:
                    # DataParallel returns losses as a vector (one per GPU) if loss is computed inside, 
                    # but since we compute loss inside the model in this architecture, 
                    # we must average the loss across the GPUs.
                    _, lm_loss = model(input_ids, position_ids=position_ids, targets=targets)
                    if isinstance(lm_loss, torch.Tensor) and lm_loss.dim() > 0:
                        lm_loss = lm_loss.mean()
                        
                    aux, z = collect_moe_losses(model)
                    total_loss = lm_loss + moe_aux_coef * aux + moe_z_coef * z
                    total_loss = total_loss / grad_accum

                if use_scaler:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                    
                step_loss   += lm_loss.item() / grad_accum
                step_tokens += input_ids.numel()

            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            if step > WARMUP_STEPS:
                scheduler.step()

            tokens_seen += step_tokens

            if _shutdown_requested:
                save_checkpoint(save_dir / "ckpt_interrupted.pt", model, optimizer, scaler, step)
                print("[INFO] Saved. Resume by re-running this script.")
                sys.exit(0)

            now = time.perf_counter()
            if step == 1 or (now - last_log_time) >= 30.0:
                current_lr = optimizer.param_groups[0]["lr"]
                print_progress(step, STEPS, step_loss, t0, tokens_seen, current_lr, gpu_cfg)
                last_log_time = now

            if step % SAVE_EVERY == 0:
                save_checkpoint(save_dir / f"ckpt_step_{step}.pt", model, optimizer, scaler, step)

    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted. Saving...")
        save_checkpoint(save_dir / "ckpt_interrupted.pt", model, optimizer, scaler, step)
        sys.exit(0)

    print("\nTraining Complete!")
    save_checkpoint(save_dir / "ckpt_final.pt", model, optimizer, scaler, STEPS)


if __name__ == "__main__":
    main()
