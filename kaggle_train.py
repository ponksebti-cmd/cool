#!/usr/bin/env python3
"""
kaggle_train.py — Kaggle Notebook Optimized Training Script (DDP + 8-bit Adam)

HOW TO USE IN KAGGLE:
═════════════════════
1. Create a new Notebook in Kaggle (GPU T4x2 accelerator).
2. Turn on Internet (Settings → Internet).
3. Paste this into a cell and run:

    !git clone https://github.com/ponksebti-cmd/cool optimistic-lavoisier
    %cd optimistic-lavoisier/optimistic-lavoisier
    !pip install -q datasets transformers bitsandbytes
    !python3 kaggle_train.py

HOW TO SAVE AND STOP CLEANLY:
══════════════════════════════
  - Press Ctrl+C  OR  click the ■ Stop button in Kaggle.
  - The script catches the signal, saves a checkpoint, then exits.
  - Checkpoints are in /kaggle/working/checkpoints/ (persist after session ends).
  - To resume: just re-run the script. It auto-loads the latest checkpoint.

MEMORY BUDGET per T4 (14.5 GB):
  Weights  FP16:  ~1.5 GB
  Grads    FP16:  ~1.5 GB
  8-bit Adam:     ~1.5 GB   (vs 6 GB standard Adam!)
  Activations:    ~4.0 GB   (seq_len=2048, grad-ckpt ON)
  Buffer:         ~2.0 GB
  Total:         ~10.5 GB ✓
"""

from __future__ import annotations
import sys
import os
import time
import signal
import threading
import queue
from contextlib import nullcontext
from pathlib import Path

# Must be set BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force stdout/stderr to flush immediately — critical for Kaggle notebook visibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

# 8-bit Adam: cuts optimizer VRAM ~6 GB → ~1.5 GB
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: run  pip install datasets transformers bitsandbytes", flush=True)
    sys.exit(1)

sys.path.insert(0, ".")
from src import Transformer, DEFAULT_300M, FLAGSHIP_700M, FLAGSHIP_1B, FLAGSHIP_3B, MAC_NANO, TINY_TEST

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STEPS           = 100_000
LR              = 4e-4
WARMUP_STEPS    = 1_000
GRAD_CLIP       = 1.0
SAVE_EVERY      = 500
LOG_EVERY_SECS  = 30.0
MODEL_CONFIG    = "700m"      # "3b" | "1b" | "700m" | "300m" | "nano" | "tiny"
KAGGLE_SAVE_DIR = "/kaggle/working/checkpoints"
LOCAL_SAVE_DIR  = "./checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def log(*args, **kwargs):
    """print() that always flushes — so Kaggle shows output immediately."""
    print(*args, **kwargs, flush=True)

def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working")

def get_save_dir() -> Path:
    if is_kaggle():
        p = Path(KAGGLE_SAVE_DIR)
        log("[KAGGLE] Kaggle environment detected.")
    else:
        p = Path(LOCAL_SAVE_DIR)
        log("[LOCAL]  Using local directory.")
    p.mkdir(parents=True, exist_ok=True)
    log(f"[SAVE]   Checkpoints → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(save_dir.glob("ckpt_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts[-1] if ckpts else None

def save_checkpoint(path: Path, model, optimizer, scaler, step: int):
    log(f"\n[SAVE] Writing {path.name} ...")
    raw = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    state = raw._orig_mod.state_dict() if hasattr(raw, "_orig_mod") else raw.state_dict()
    obj = {
        "model_state_dict":     state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step":                 step,
    }
    if scaler:
        obj["scaler_state_dict"] = scaler.state_dict()
    torch.save(obj, path)
    log(f"[SAVE] ✓ Checkpoint saved at step {step} → {path.name}")

def load_checkpoint(path: Path, model, optimizer, scaler, dev) -> int:
    log(f"[RESUME] Loading {path.name} ...")
    ckpt = torch.load(path, map_location=dev)
    raw = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    raw.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt["step"] + 1
    log(f"[RESUME] ✓ Resuming from step {step}")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# DATA — background thread so GPU never idles waiting for tokens
# ─────────────────────────────────────────────────────────────────────────────
def _data_producer(tokenizer, batch_size: int, block_size: int,
                   out_q: "queue.Queue[dict]", stop_evt: threading.Event,
                   rank: int = 0, world_size: int = 1):
    """Runs in a background thread. Fills a queue with pre-tokenized batches."""
    eos = tokenizer.eos_token_id or 0
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True
    )
    # Each rank skips to a different starting shard so they don't duplicate data
    if world_size > 1:
        dataset = dataset.skip(rank * 50_000)

    buf: list[int] = []
    need = batch_size * block_size
    for row in dataset:
        if stop_evt.is_set():
            break
        try:
            buf.extend(tokenizer.encode(row["text"], add_special_tokens=True))
        except Exception:
            continue
        while len(buf) >= need and not stop_evt.is_set():
            chunk = buf[:need]
            buf   = buf[need:]
            ids   = torch.tensor(chunk, dtype=torch.long).view(batch_size, block_size)
            tgts  = torch.empty_like(ids)
            tgts[:, :-1] = ids[:, 1:]
            tgts[:, -1]  = eos
            pos   = torch.arange(block_size).unsqueeze(0).expand(batch_size, -1)
            while not stop_evt.is_set():
                try:
                    out_q.put(
                        {"input_ids": ids, "targets": tgts, "position_ids": pos},
                        timeout=1.0,
                    )
                    break
                except queue.Full:
                    pass

def make_data_loader(tokenizer, batch_size: int, block_size: int,
                     rank: int = 0, world_size: int = 1):
    q        = queue.Queue(maxsize=8)
    stop_evt = threading.Event()
    t = threading.Thread(
        target=_data_producer,
        args=(tokenizer, batch_size, block_size, q, stop_evt, rank, world_size),
        daemon=True,
    )
    t.start()
    def _gen():
        while True:
            yield q.get()
    return _gen(), stop_evt


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def collect_moe_losses(model):
    raw = model.module if isinstance(model, DDP) else model
    aux = z = 0.0
    for layer in raw.layers:
        mlp = layer.mlp
        if hasattr(mlp, "l_aux"):
            v = mlp.l_aux
            aux += v.item() if isinstance(v, torch.Tensor) else float(v)
        if hasattr(mlp, "l_z"):
            v = mlp.l_z
            z   += v.item() if isinstance(v, torch.Tensor) else float(v)
    return aux, z

def fmt_toks(n: int) -> str:
    return f"{n/1e9:.3f}B" if n >= 1e9 else f"{n/1e6:.1f}M"

def print_progress(step, total, loss, t0, tokens_seen, lr, vram_gb):
    elapsed = time.perf_counter() - t0
    tok_s   = tokens_seen / elapsed if elapsed > 0 else 0
    pct     = step / total
    eta_s   = (elapsed / pct - elapsed) if pct > 0 else 0
    h, rem  = divmod(int(eta_s), 3600)
    m, _s   = divmod(rem, 60)
    bar     = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
    log(
        f"  [{bar}] {pct*100:3.0f}% | "
        f"Step: {step:,}/{total:,} | "
        f"Toks: {fmt_toks(tokens_seen)} | "
        f"Loss: {loss:.4f} | "
        f"Tok/s: {tok_s:,.0f} | "
        f"LR: {lr:.2e} | "
        f"ETA: {h}h {m:02d}m | "
        f"VRAM: {vram_gb:.1f}G"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
    name = "Ctrl+C" if sig == signal.SIGINT else "Stop button (SIGTERM)"
    log(f"\n[STOP] {name} received — saving after this step...\n")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING WORKER — one process per GPU
# ─────────────────────────────────────────────────────────────────────────────
def train_worker(rank: int, world_size: int, save_dir: Path, amp_dtype: torch.dtype):
    is_master = (rank == 0)

    # ── DDP init ────────────────────────────────────────────────────────────
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    use_scaler = (amp_dtype == torch.float16)
    torch.backends.cudnn.benchmark = True
    is_bf16 = (amp_dtype == torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = is_bf16
    torch.backends.cudnn.allow_tf32 = is_bf16

    # ── Model config ─────────────────────────────────────────────────────────
    cfg_map = {
        "tiny": TINY_TEST, "nano": MAC_NANO, "700m": FLAGSHIP_700M,
        "1b": FLAGSHIP_1B, "3b": FLAGSHIP_3B,
    }
    config = cfg_map.get(MODEL_CONFIG, DEFAULT_300M)
    config.use_gradient_checkpointing = True

    # ── Build model ──────────────────────────────────────────────────────────
    if is_master:
        log("\n[MODEL] Building model ...")
    model = Transformer(config).to(dev)
    param_count = sum(p.numel() for p in model.parameters())
    if is_master:
        log(f"[MODEL] {param_count/1e6:.1f}M parameters")

    if world_size > 1:
        model = DDP(
            model, device_ids=[rank], output_device=rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=LR, betas=(0.9, 0.95),
            weight_decay=0.1, eps=1e-8,
        )
        opt_label = "AdamW 8-bit (bitsandbytes) — saves ~4.5 GB VRAM"
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, betas=(0.9, 0.95),
            weight_decay=0.1, eps=1e-8, fused=True,
        )
        opt_label = "AdamW FP32 (install bitsandbytes to save 4.5 GB VRAM)"

    scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR * 0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── Auto-resume ──────────────────────────────────────────────────────────
    start_step = 1
    if is_master:
        ckpt = find_latest_checkpoint(save_dir)
        if ckpt:
            start_step = load_checkpoint(ckpt, model, optimizer, scaler, dev)
            for i in range(start_step - 1):
                if i >= WARMUP_STEPS:
                    scheduler.step()

    if world_size > 1:
        t = torch.tensor([start_step], dtype=torch.long, device=dev)
        dist.broadcast(t, src=0)
        start_step = int(t.item())

    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)

    batch_size_per_gpu = 2
    grad_accum = max(1, 32 // (batch_size_per_gpu * world_size))

    moe_aux_c = getattr(config, "moe_aux_loss_coef", 0.0)
    moe_z_c   = getattr(config, "moe_z_loss_coef",   0.0)

    # ── Banner ───────────────────────────────────────────────────────────────
    if is_master:
        gpu_name  = torch.cuda.get_device_name(0).upper()
        prec_name = "fp16 ✦ Tensor Cores" if use_scaler else "bfloat16"
        eff_batch  = batch_size_per_gpu * world_size * grad_accum
        log(f"\n{'='*62}")
        log(f"  KAGGLE TRAINING  ──  DDP + 8-bit Adam")
        log(f"{'='*62}")
        log(f"  GPUs        : {world_size}x {gpu_name}")
        log(f"  Parameters  : {param_count/1e6:.1f}M")
        log(f"  Optimizer   : {opt_label}")
        log(f"  Batching    : {batch_size_per_gpu*world_size}/step  ×  {grad_accum} accum = {eff_batch} effective")
        log(f"  Seq length  : {config.block_size} tokens")
        log(f"  Precision   : {prec_name}")
        log(f"  GradScaler  : {'✓' if use_scaler else 'off'}")
        log(f"  Checkpoints : {save_dir}  (every {SAVE_EVERY} steps)")
        log(f"  Resuming    : step {start_step}/{STEPS}")
        log(f"{'='*62}")
        log(f"\n  ► STOP & SAVE: press Ctrl+C  or  click ■ in Kaggle")
        log(f"    Script saves a checkpoint then exits cleanly.\n")
        log("[DATA] Connecting to FineWeb-Edu 10B ...")

    # ── Data ─────────────────────────────────────────────────────────────────
    tokenizer  = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    batch_iter, stop_evt = make_data_loader(
        tokenizer, batch_size_per_gpu, config.block_size,
        rank=rank, world_size=world_size,
    )
    if is_master:
        log("[DATA] ✓ Data loader started — training begins!\n")

    # ── Signals (rank 0 only to avoid duplicate saves) ────────────────────────
    if is_master:
        signal.signal(signal.SIGINT,  _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    model.train()
    tokens_seen = 0
    t0          = time.perf_counter()
    last_log    = t0 - LOG_EVERY_SECS  # Force log on step 1
    step_loss   = 0.0

    try:
        for step in range(start_step, STEPS + 1):

            # Warmup LR
            if step <= WARMUP_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step / max(1, WARMUP_STEPS))

            optimizer.zero_grad(set_to_none=True)
            step_loss   = 0.0
            step_tokens = 0

            for acc_i in range(grad_accum):
                batch = next(batch_iter)
                ids   = batch["input_ids"].to(dev, non_blocking=True)
                tgts  = batch["targets"].to(dev, non_blocking=True)
                pids  = batch["position_ids"].to(dev, non_blocking=True)

                # Skip cross-GPU sync on intermediate accumulation steps
                sync_ctx = nullcontext() if (world_size == 1 or acc_i == grad_accum - 1) \
                           else model.no_sync()

                with sync_ctx, amp_ctx:
                    _, lm_loss = model(ids, position_ids=pids, targets=tgts)
                    if isinstance(lm_loss, torch.Tensor) and lm_loss.dim() > 0:
                        lm_loss = lm_loss.mean()
                    aux, z     = collect_moe_losses(model)
                    total_loss = (lm_loss + moe_aux_c * aux + moe_z_c * z) / grad_accum

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

            # ── Graceful stop ─────────────────────────────────────────────────
            if _shutdown:
                if is_master:
                    save_checkpoint(save_dir / f"ckpt_step_{step}.pt",
                                    model, optimizer, scaler, step)
                    log(f"\n[STOP] ✓ Saved at step {step}. Re-run to resume.\n")
                break

            # ── Log every 30s ─────────────────────────────────────────────────
            now = time.perf_counter()
            if is_master and (now - last_log) >= LOG_EVERY_SECS:
                vram = torch.cuda.memory_allocated(0) / 1e9
                print_progress(step, STEPS, step_loss, t0, tokens_seen,
                               optimizer.param_groups[0]["lr"], vram)
                last_log = now

            # ── Periodic checkpoint ───────────────────────────────────────────
            if is_master and step % SAVE_EVERY == 0:
                save_checkpoint(save_dir / f"ckpt_step_{step}.pt",
                                model, optimizer, scaler, step)

        # ── Finished ─────────────────────────────────────────────────────────
        if is_master and not _shutdown:
            log("\n[DONE] Training complete!")
            save_checkpoint(save_dir / "ckpt_final.pt", model, optimizer, scaler, STEPS)

    except Exception as e:
        # Always attempt to save on unexpected crash
        if is_master:
            log(f"\n[ERROR] Unexpected crash: {e}")
            log("[ERROR] Attempting emergency checkpoint save ...")
            try:
                save_checkpoint(save_dir / f"ckpt_crash_step_{step}.pt",
                                model, optimizer, scaler, step)
            except Exception as save_err:
                log(f"[ERROR] Emergency save also failed: {save_err}")
        raise

    finally:
        # Always clean up — prevents the 'destroy_process_group' warning
        stop_evt.set()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    save_dir = get_save_dir()

    if not torch.cuda.is_available():
        log("[WARN] No GPU — CPU only (very slow).")
        train_worker(0, 1, save_dir, torch.float32)
        return

    world_size = torch.cuda.device_count()
    gpu_name   = torch.cuda.get_device_name(0).upper()
    is_ampere  = any(x in gpu_name for x in ["A100", "A10", "L4", "H100", "RTX 4", "RTX 3"])
    amp_dtype  = torch.bfloat16 if is_ampere else torch.float16

    if world_size > 1:
        log(f"[DDP] Launching {world_size}-GPU DDP training ...")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        mp.spawn(
            train_worker,
            args=(world_size, save_dir, amp_dtype),
            nprocs=world_size,
            join=True,
        )
    else:
        log(f"[SINGLE-GPU] {gpu_name}")
        train_worker(0, 1, save_dir, amp_dtype)


if __name__ == "__main__":
    main()
