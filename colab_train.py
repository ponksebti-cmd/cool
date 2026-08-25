#!/usr/bin/env python3
"""
colab_train.py — Google Colab Optimized Training Script

HOW TO USE IN COLAB:
════════════════════
Paste this into a Colab cell and run it:

    # Cell 1: Clone your repo and install deps
    !git clone https://github.com/YOUR_USER/YOUR_REPO optimistic-lavoisier
    %cd optimistic-lavoisier
    !pip install -q datasets transformers

    # Cell 2: Run training (checkpoints auto-save to Google Drive)
    !python3 colab_train.py

FEATURES:
  ✓ Auto-detects GPU (T4/V100/A100/L4) and selects best dtype
  ✓ Auto-mounts Google Drive and saves checkpoints there
  ✓ Auto-resumes from latest Drive checkpoint on reconnect
  ✓ Keeps Colab session alive with a JavaScript keepalive hack
  ✓ Streams MiniPile dataset (tiny shards, fast startup)
  ✓ GradScaler for FP16 (T4/V100 Tensor Core safety)
  ✓ Graceful Ctrl+C saving
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

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("\n[ERROR] Missing: pip install datasets transformers\n")
    sys.exit(1)

sys.path.insert(0, ".")
from src import Transformer, DEFAULT_300M, MAC_NANO, TINY_TEST

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these to your needs
# ─────────────────────────────────────────────────────────────────────────────
STEPS          = 100_000
LR             = 4e-4
WARMUP_STEPS   = 1_000
GRAD_CLIP      = 1.0
LOG_EVERY      = 10
SAVE_EVERY     = 500        # save to Drive every 500 steps (~10 min on T4)
MODEL_CONFIG   = "300m"     # "300m" | "nano" | "tiny"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/ai_checkpoints"
LOCAL_SAVE_DIR = "/content/ai_checkpoints"   # fallback if Drive not mounted


# ─────────────────────────────────────────────────────────────────────────────
# COLAB DETECTION & DRIVE MOUNT
# ─────────────────────────────────────────────────────────────────────────────
def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive() -> Path:
    """Mount Google Drive and return the checkpoint directory path."""
    if is_colab():
        try:
            from google.colab import drive
            print("[DRIVE] Mounting Google Drive...")
            drive.mount("/content/drive", force_remount=False)
            save_dir = Path(DRIVE_SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DRIVE] ✓ Checkpoints will be saved to: {save_dir}")
            return save_dir
        except Exception as e:
            print(f"[DRIVE] ⚠ Could not mount Drive ({e}). Using local /content/")
    
    save_dir = Path(LOCAL_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SAVE]  Checkpoints will be saved to: {save_dir}")
    return save_dir


def colab_keepalive():
    """
    Inject a JavaScript snippet into the Colab notebook to click the page
    every 60 seconds, preventing the 90-minute idle timeout.
    Only runs when inside an actual Colab session.
    """
    if not is_colab():
        return
    try:
        from IPython.display import display, Javascript
        js = """
        function ClickConnect(){
            console.log("Keepalive: preventing Colab idle timeout...");
            document.querySelector("colab-toolbar-button#connect").click();
        }
        setInterval(ClickConnect, 60000);
        """
        display(Javascript(js))
        print("[KEEPALIVE] ✓ Session keepalive active (clicks every 60s to prevent timeout)")
    except Exception:
        pass  # Not in notebook context, skip silently


# ─────────────────────────────────────────────────────────────────────────────
# GPU AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_gpu_config() -> dict:
    """
    Detect the GPU and return optimal training settings.
    
    T4  (Turing,  16GB) → FP16 Tensor Cores, batch=4
    V100 (Volta,  16GB) → FP16 Tensor Cores, batch=4
    A100 (Ampere, 40GB) → BF16 Tensor Cores, batch=16
    L4   (Ada,    24GB) → BF16 Tensor Cores, batch=8
    """
    if not torch.cuda.is_available():
        return {"device": "cpu", "dtype": torch.float32, "batch_size": 1,
                "grad_accum": 4, "use_scaler": False, "name": "CPU"}
    
    name = torch.cuda.get_device_name(0).upper()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Ampere+ (A100, A10, RTX 3000+) and Ada (L4, RTX 4000+) support BF16 Tensor Cores
    is_ampere_plus = any(x in name for x in ["A100", "A10G", "A10 ", "L4", "L40", "H100", "RTX 4", "RTX 3"])
    
    if is_ampere_plus:
        dtype = torch.bfloat16
        use_scaler = False  # BF16 doesn't need GradScaler
        batch_size = 16 if vram_gb > 35 else 8
    else:
        # T4, V100, K80 — FP16 Tensor Cores
        dtype = torch.float16
        use_scaler = True   # FP16 REQUIRES GradScaler
        batch_size = 4
    
    return {
        "device": "cuda",
        "dtype": dtype,
        "batch_size": batch_size,
        "grad_accum": max(1, 32 // batch_size),  # target effective batch=32
        "use_scaler": use_scaler,
        "name": name,
        "vram_gb": vram_gb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(save_dir: Path) -> Path | None:
    """Search for the most recent checkpoint file in the save directory."""
    checkpoints = sorted(save_dir.glob("ckpt_step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(path: Path, model, optimizer, scaler, step: int):
    sys.stdout.write("\n")
    print(f"[SAVE] Saving to {path} ...")
    state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
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
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_step = ckpt["step"] + 1
    print(f"[RESUME] ✓ Resuming from step {start_step}")
    return start_step


# ─────────────────────────────────────────────────────────────────────────────
# DATA STREAMING
# ─────────────────────────────────────────────────────────────────────────────
def stream_packed_batches(tokenizer, batch_size: int, block_size: int) -> Iterator:
    dataset = load_dataset("JeanKaddour/minipile", split="train", streaming=True)
    buffer = []
    for row in dataset:
        tokens = tokenizer.encode(row["text"], add_special_tokens=True)
        buffer.extend(tokens)
        tokens_needed = batch_size * block_size
        while len(buffer) >= tokens_needed:
            chunk = buffer[:tokens_needed]
            buffer = buffer[tokens_needed:]
            input_ids = torch.tensor(chunk, dtype=torch.long).view(batch_size, block_size)
            targets = torch.empty_like(input_ids)
            targets[:, :-1] = input_ids[:, 1:]
            targets[:, -1] = tokenizer.eos_token_id or 0
            pos_ids = torch.arange(block_size).unsqueeze(0).expand(batch_size, -1)
            yield {"input_ids": input_ids, "targets": targets, "position_ids": pos_ids}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def collect_moe_losses(model):
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


def print_progress(step, total, loss, t0, tokens_seen, lr, gpu_cfg):
    elapsed = time.perf_counter() - t0
    tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
    percent = step / total
    eta_s = (elapsed / percent - elapsed) if percent > 0 else 0
    h, rem = divmod(int(eta_s), 3600)
    m, s = divmod(rem, 60)
    eta_str = f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s"
    bar = "█" * int(25 * percent) + "░" * (25 - int(25 * percent))
    vram = f" | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else ""
    sys.stdout.write(
        f"\r  [{bar}] {percent*100:3.0f}% | Loss: {loss:.4f} | "
        f"Tok/s: {tok_per_sec:,.0f} | LR: {lr:.2e} | ETA: {eta_str}{vram}   "
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
    # 1. Mount Drive and set up session keepalive
    save_dir = mount_drive()
    colab_keepalive()

    # 2. Detect GPU and configure automatically
    gpu_cfg = detect_gpu_config()
    dev = torch.device(gpu_cfg["device"])
    amp_dtype = gpu_cfg["dtype"]
    batch_size = gpu_cfg["batch_size"]
    grad_accum = gpu_cfg["grad_accum"]
    use_scaler = gpu_cfg["use_scaler"]

    # 3. CUDA backend tuning
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # TF32 only benefits Ampere+; disable on T4 to prevent silent precision loss
        is_ampere = amp_dtype == torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = is_ampere
        torch.backends.cudnn.allow_tf32 = is_ampere

    # 4. Load model config
    if MODEL_CONFIG == "tiny":
        config = TINY_TEST
    elif MODEL_CONFIG == "nano":
        config = MAC_NANO
    else:
        config = DEFAULT_300M
    config.use_gradient_checkpointing = True

    # 5. Build model
    print("\nBuilding model...")
    model = Transformer(config).to(dev)
    param_count = sum(p.numel() for p in model.parameters())

    # 6. Optimizer (fused on CUDA = much faster)
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95),
        weight_decay=0.1, eps=1e-8,
        fused=(dev.type == "cuda")
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # 7. Auto-resume from latest Drive checkpoint
    start_step = 1
    latest_ckpt = find_latest_checkpoint(save_dir)
    if latest_ckpt:
        start_step = load_checkpoint(latest_ckpt, model, optimizer, scaler, dev)
        # Fast-forward scheduler to match
        for _ in range(start_step - 1):
            if _ > WARMUP_STEPS:
                scheduler.step()
    
    # 8. AMP context
    use_amp = (dev.type in ("cuda", "mps"))
    amp_ctx = torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp) if use_amp else nullcontext()

    # 9. Data pipeline
    dtype_name = "fp16 ✦ Tensor Cores" if amp_dtype == torch.float16 else "bfloat16 ✦ Tensor Cores"
    print(f"\n{'='*62}")
    print(f"  COLAB TRAINING  ──  GPU OPTIMIZED")
    print(f"{'='*62}")
    print(f"  GPU         : {gpu_cfg.get('name', 'N/A')} ({gpu_cfg.get('vram_gb', 0):.0f} GB)")
    print(f"  Parameters  : {param_count/1e6:.1f}M")
    print(f"  Micro-batch : {batch_size}  ×  {grad_accum} accum = {batch_size*grad_accum} effective")
    print(f"  AMP         : {dtype_name}")
    print(f"  GradScaler  : {'✓' if use_scaler else 'off (BF16 does not need it)'}")
    print(f"  Checkpoints : {save_dir}")
    print(f"  Resuming    : step {start_step}")
    print(f"{'='*62}\n")

    print("[DATA] Connecting to dataset (MiniPile)...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    batch_iter = stream_packed_batches(tokenizer, batch_size, config.block_size)

    signal.signal(signal.SIGINT, _sigint_handler)
    
    moe_aux_coef = getattr(config, "moe_aux_loss_coef", 0.0)
    moe_z_coef   = getattr(config, "moe_z_loss_coef",   0.0)

    model.train()
    tokens_seen = 0
    t0 = time.perf_counter()

    try:
        for step in range(start_step, STEPS + 1):

            # LR warmup
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
                    _, lm_loss = model(input_ids, position_ids=position_ids, targets=targets)
                    aux, z = collect_moe_losses(model)
                    total_loss = lm_loss + moe_aux_coef * aux + moe_z_coef * z
                    total_loss = total_loss / grad_accum

                scaler.scale(total_loss).backward()
                step_loss   += lm_loss.item() / grad_accum
                step_tokens += input_ids.numel()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            if step > WARMUP_STEPS:
                scheduler.step()

            tokens_seen += step_tokens

            # Shutdown check
            if _shutdown_requested:
                save_checkpoint(save_dir / "ckpt_interrupted.pt", model, optimizer, scaler, step)
                print("[INFO] Saved. Resume by re-running this script.")
                sys.exit(0)

            # Logging
            if step == 1 or step % LOG_EVERY == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print_progress(step, STEPS, step_loss, t0, tokens_seen, current_lr, gpu_cfg)

            # Checkpoint to Drive
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
