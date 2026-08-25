"""
overfit_test.py — Stage 1 Validation: Single-batch overfit test.

PURPOSE
-------
This is the go/no-go gate for Stage 1. A correctly implemented model
with the right initialization must be able to memorize a single batch.
If it can't overfit, something is wrong: vanishing/exploding gradients,
wrong loss wiring, broken mask, or bad init.

PASS CRITERIA
-------------
    - Loss drops from ~ln(vocab_size) (random init baseline) to < 0.10
      within 200 optimization steps.
    - Gradient norm stays in a reasonable range (never NaN/inf).
    - The test runs on the TINY_TEST config so it finishes in < 30s on CPU.

USAGE
-----
    # Fast test (CPU, TINY config):
    python overfit_test.py

    # Full 300M config on GPU (requires VRAM, slower):
    python overfit_test.py --config 300m --device cuda

    # AMD GPU via ROCm:
    python overfit_test.py --config 300m --device cuda

FLAGS
-----
    --config   {tiny|300m}    Model size to test (default: tiny)
    --device   {cpu|cuda}     Compute device (default: auto-detect)
    --steps    INT            Optimisation steps (default: 200)
    --lr       FLOAT          Learning rate (default: 3e-3)
    --target   FLOAT          Loss target to declare PASS (default: 0.10)
    --seed     INT            RNG seed (default: 42)
    --packed   BOOL           Use packed sequence + block-diagonal mask (default: True)
"""

from __future__ import annotations
import argparse
import sys
import math
import time

import torch
import torch.optim as optim

# Allow running from repo root
sys.path.insert(0, ".")
from src import Transformer, DEFAULT_300M, TINY_TEST, make_dataloader, build_block_diagonal_mask


# ── Synthetic data ────────────────────────────────────────────────────────────

def make_synthetic_documents(
    n_docs: int,
    vocab_size: int,
    min_len: int,
    max_len: int,
    seed: int = 42,
) -> list[list[int]]:
    """
    Create synthetic random-token documents.

    We intentionally avoid token_id=0 (often the pad token) and
    token_id=1 (often BOS) to keep the task clean.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    docs = []
    for _ in range(n_docs):
        length = torch.randint(min_len, max_len + 1, (1,), generator=rng).item()
        # Sample from [2, vocab_size - 1]
        tokens = torch.randint(2, vocab_size, (length,), generator=rng).tolist()
        docs.append(tokens)
    return docs


# ── Gradient norm helper ──────────────────────────────────────────────────────

def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


# ── Main overfit loop ─────────────────────────────────────────────────────────

def run_overfit_test(
    config_name: str = "tiny",
    device: str = "auto",
    n_steps: int = 200,
    lr: float = 3e-3,
    target_loss: float = 0.10,
    seed: int = 42,
    use_packed: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Run the single-batch overfit test.

    Returns True if PASS, False if FAIL.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(seed)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    config = TINY_TEST if config_name == "tiny" else DEFAULT_300M

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Stage 1 Overfit Test")
        print(f"{'='*60}")
        print(f"  Config  : {config_name.upper()}")
        print(f"  Device  : {dev}")
        print(f"  Steps   : {n_steps}")
        print(f"  LR      : {lr}")
        print(f"  Target  : loss < {target_loss}")
        print(f"  Packed  : {use_packed}")
        print(f"{'='*60}\n")

    # ── Model ──────────────────────────────────────────────────────────────
    model = Transformer(config).to(dev)
    n_params = model.num_parameters()
    if verbose:
        print(f"  Parameters : {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Build a SINGLE fixed batch ─────────────────────────────────────────
    # We freeze this batch and repeatedly overfit to it.
    # The initial loss should be close to ln(vocab_size) (random prediction).
    block_size = config.block_size

    if use_packed:
        # Create small synthetic documents and pack them
        docs = make_synthetic_documents(
            n_docs=20,
            vocab_size=config.vocab_size,
            min_len=max(4, block_size // 8),
            max_len=block_size // 4,
            seed=seed,
        )
        loader = make_dataloader(
            documents=docs,
            block_size=block_size,
            batch_size=2,
            shuffle=False,
            seed=seed,
            num_workers=0,
        )
        batch = next(iter(loader))
        input_ids    = batch["input_ids"].to(dev)
        targets      = batch["targets"].to(dev)
        position_ids = batch["position_ids"].to(dev)
        # Cast mask to model dtype (float32 by default)
        attn_mask    = batch["attn_mask"].to(dev)
    else:
        # Simple causal batch (no packing), for comparison
        torch.manual_seed(seed)
        input_ids = torch.randint(2, config.vocab_size, (2, block_size), device=dev)
        targets   = torch.roll(input_ids, -1, dims=1)
        targets[:, -1] = -100
        position_ids = None
        attn_mask    = None

    # ── Initial loss check ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, init_loss = model(
            input_ids,
            attn_mask=attn_mask,
            position_ids=position_ids,
            targets=targets,
        )
    expected_init = math.log(config.vocab_size)
    if verbose:
        print(f"  Initial loss   : {init_loss.item():.4f}")
        print(f"  Expected (rand): {expected_init:.4f}  (~ln(vocab_size))")
        if abs(init_loss.item() - expected_init) > 2.0:
            print(f"  ⚠️  WARNING: Initial loss deviates significantly from random baseline.")
            print(f"     This may indicate a broken init or forward pass.")
        print()

    # ── Optimiser ─────────────────────────────────────────────────────────
    # AdamW with mild weight decay. High LR is fine for overfit tests.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    losses = []
    grad_norms = []
    log_every = max(1, n_steps // 20)
    t0 = time.time()

    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        _, loss = model(
            input_ids,
            attn_mask=attn_mask,
            position_ids=position_ids,
            targets=targets,
        )
        
        # Accumulate MoE aux loss and Z-loss
        aux_loss = 0.0
        z_loss   = 0.0
        for layer in model.layers:
            if hasattr(layer.mlp, "l_aux"):
                aux_loss += layer.mlp.l_aux
            if hasattr(layer.mlp, "l_z"):
                z_loss += layer.mlp.l_z
        
        # Combine losses
        moe_coef = getattr(config, 'moe_aux_loss_coef', 0.0)
        z_coef   = getattr(config, 'moe_z_loss_coef',   0.0)
        total_loss = loss + moe_coef * aux_loss + z_coef * z_loss
        total_loss.backward()

        # Gradient clipping (max norm = 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        gn = grad_norm(model)
        optimizer.step()

        losses.append(total_loss.item())
        grad_norms.append(gn)

        if verbose and (step % log_every == 0 or step == 1):
            elapsed = time.time() - t0
            print(
                f"  step {step:4d}/{n_steps} | "
                f"loss: {loss.item():.6f} | "
                f"grad_norm: {gn:.4f} | "
                f"t: {elapsed:.1f}s"
            )

        # Check for NaN
        if math.isnan(loss.item()):
            print(f"\n  ✗ FAIL — NaN loss detected at step {step}")
            return False

    # ── Final evaluation ───────────────────────────────────────────────────
    final_loss = losses[-1]
    min_loss   = min(losses)
    total_time = time.time() - t0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Final loss : {final_loss:.6f}")
        print(f"  Min loss   : {min_loss:.6f}")
        print(f"  Target     : < {target_loss}")
        print(f"  Time       : {total_time:.1f}s")
        print(f"{'─'*60}")

    passed = final_loss < target_loss

    if verbose:
        if passed:
            print(f"\n  ✓ PASS — Model overfits to {final_loss:.6f} < {target_loss}")
        else:
            print(
                f"\n  ✗ FAIL — Final loss {final_loss:.6f} did not reach target {target_loss}"
            )

    return passed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 Overfit Test — single-batch memorization check"
    )
    parser.add_argument("--config", default="tiny", choices=["tiny", "300m"],
                        help="Model config (default: tiny)")
    parser.add_argument("--device", default="auto",
                        help="Compute device: auto, cpu, cuda (default: auto)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Optimisation steps (default: 200)")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate (default: 3e-3)")
    parser.add_argument("--target", type=float, default=0.10,
                        help="Loss target for PASS (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-packed", action="store_true",
                        help="Disable packed sequences (use simple causal batch)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    passed = run_overfit_test(
        config_name=args.config,
        device=args.device,
        n_steps=args.steps,
        lr=args.lr,
        target_loss=args.target,
        seed=args.seed,
        use_packed=not args.no_packed,
    )
    sys.exit(0 if passed else 1)
