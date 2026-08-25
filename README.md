# Production LM — Stage 1: Dense Transformer Baseline

> **Target**: ~295M parameters, optimized for AMD RX 6650 XT (ROCm)

## Architecture

| Component | Choice | Notes |
|---|---|---|
| Normalization | RMSNorm (pre-norm) | Computed in fp32, cast back to input dtype |
| Attention | GQA — 16Q / 4KV heads | `repeat_kv` avoids data copies |
| Position encoding | RoPE | Shared cache across all layers |
| FFN | SwiGLU | `ffn_dim = round(8/3 × hidden_dim, 64)` |
| Embedding tying | ✅ | Saves ~33M params |
| Residual init | Scaled — `1/√(2L)` | Prevents exploding variance at depth |

### 300M Config

```
hidden_dim  = 1024
n_layers    = 24
n_heads     = 16
n_kv_heads  = 4
head_dim    = 64
ffn_dim     = 2752
max_seq_len = 2048
vocab_size  = 32768 (padded to ×64)
Params      ≈ 295M (with tied embeddings)
```

## File Layout

```
src/
├── __init__.py      — Public API
├── config.py        — ModelConfig dataclass + presets
├── rope.py          — Rotary Position Embedding
├── attention.py     — Grouped-Query Attention
├── mlp.py           — SwiGLU FFN
└── model.py         — Transformer + RMSNorm + TransformerBlock

overfit_test.py      — Stage 1 validation (go/no-go gate)
train.py             — Reference training loop
tests/
└── test_stage1.py   — 10 test classes, 25+ unit tests
requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Run all unit tests + overfit validation
pytest tests/test_stage1.py -v

# Run just the overfit gate (TINY config, CPU, ~20s)
python overfit_test.py

# Run on GPU with 300M config
python overfit_test.py --config 300m --device cuda

# Reference training run
python train.py --config tiny --steps 200
```

## AMD RX 6650 XT Notes

The RX 6650 XT (RDNA2, 8GB VRAM) requires **ROCm** PyTorch:

```bash
# Install ROCm PyTorch (replace rocm5.7 with your ROCm version)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

Key constraints for this GPU:
- **VRAM**: 8GB → batch_size × seq_len must fit within ~6GB (leave headroom for activations)
- **bf16**: Supported natively on RDNA2 → use `--dtype bf16` in Stage 5
- **Flash Attention**: Not natively available; we use `F.scaled_dot_product_attention`
  which dispatches to an efficient math kernel on ROCm 6.0+
- **Memory estimate** (300M, batch=1, seq=2048, fp32): ~4.5GB

## Stage Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| **1** | Dense Transformer + data pipeline | ✅ |
| **2** | Hybrid attention (3:1 linear/full) | ✅ |
| **3** | Sparse MoE (8 experts, top-2 + shared) | ✅ |
| **4** | Multi-token prediction heads | ✅ **Current** |
| 5 | bf16/fp8 + FSDP | — |

## Hybrid Architecture (Stage 2)

We implement a 3:1 pattern out-of-the-box (`attn_layer_period = 4`). 
- 1 in 4 layers uses `GroupedQueryAttention` (Dense).
- 3 in 4 layers use `GatedDeltaNet` (Linear Attention).
The DeltaNet is a pure PyTorch implementation leveraging `cumsum` for the recurrent state, avoiding custom Triton/CUDA kernels while maintaining O(T²) training speed (perfectly scalable for T ≤ 2048).

## Sparse Mixture of Experts (Stage 3)

The standard MLP block has been swapped out for a fully featured `MoELayer`:
- 8 independent experts + 1 always-active shared expert.
- Top-2 Routing mechanism.
- Capacity dropping factor (`1.25x`) limits token congestion per expert.
- Standard Switch Transformer auxiliary load-balancing loss (`L_aux`).

## Multi-Token Prediction (Stage 4)

We've added independent MTP (Multi-Token Prediction) heads to predict future tokens simultaneously:
- Predicts `num_mtp_heads=2` extra future tokens (t+1, t+2, t+3).
- Each head uses an independent `SiLU` MLP projection mapped to the tied language modeling head.
- Modifies the data pipeline (`PackedDataset`) to yield a multi-dimensional shifted target array with comprehensive cross-document masking for all `K` steps.
- Configurable loss weighting for the MTP targets vs the base target.

## Validation (Stage 1-4 Gate)

The overfit test must pass before Stage 2:

```
✓ PASS — Model overfits to X.XXXXXX < 0.10
```

Pass criteria:
1. Loss from `~ln(vocab_size)` → `< 0.10` within 200 AdamW steps
2. No NaN gradients at any step
3. Both packed (block-diagonal mask) and simple causal modes pass
