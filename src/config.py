"""
config.py — Central configuration for the transformer model.

All hyperparameters are defined here. Downstream code imports this
dataclass and does NOT hard-code any numbers directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class ModelConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────
    vocab_size: int = 32_768          # power-of-2 friendly; pad to multiple of 64
    pad_to_multiple: int = 64         # keeps matmuls on 64-byte boundaries

    # ── Dimensionality ────────────────────────────────────────────────────
    hidden_dim: int = 1024            # d_model
    n_layers: int = 24                # transformer depth
    n_heads: int = 16                 # query heads
    n_kv_heads: int = 4              # key/value heads (GQA); must divide n_heads evenly
    head_dim: int = 64               # = hidden_dim // n_heads (auto-derived below)

    # ── Feed-forward (SwiGLU) ─────────────────────────────────────────────
    # SwiGLU effective expansion: we split the FFN into gate+up projections,
    # so true width = ffn_dim, but weight count is 2× ffn_dim.
    # Standard practice: ffn_dim = int(8/3 * hidden_dim), rounded to multiple of 64.
    ffn_dim: int = field(init=False)

    # ── RoPE ─────────────────────────────────────────────────────────────
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    # YaRN scaling: factor=1.0 is identical to standard RoPE.
    # Set >1 to extend effective context with no extra parameters.
    rope_scaling_factor: float = 1.0
    # 'yarn' applies non-uniform interpolation; 'linear' is naive linear.
    rope_scaling_type: str = "yarn"

    # ── Regularisation ────────────────────────────────────────────────────
    dropout: float = 0.0              # 0 for inference; small value for training

    # ── Initialization ────────────────────────────────────────────────────
    # Wang et al. (2022) "DeepNet" / GPT-NeoX style scaled init.
    # Residual projections are multiplied by 1/sqrt(2*n_layers) at init.
    init_std: float = 0.02
    scale_residual_init: bool = True  # apply 1/sqrt(2L) scaling to o_proj & down_proj

    # ── Hybrid Architecture ───────────────────────────────────────────────
    # We use a 3:1 hybrid pattern by default.
    # If layer_idx % attn_layer_period == (attn_layer_period - 1), we use Dense GQA.
    # Otherwise, we use Linear Attention (Gated DeltaNet).
    attn_layer_period: int = 4

    # ── Sparse MoE ────────────────────────────────────────────────────────
    n_routed_experts: int = 0             # Set >0 to enable MoE
    n_active_experts: int = 2
    use_shared_expert: bool = False
    moe_capacity_factor: float = 1.25     # > 0 enforces a max token limit per expert
    moe_aux_loss_coef: float = 0.01       # typical coefficient for load balancing
    moe_z_loss_coef: float = 1e-3         # Z-loss: penalises large router logits (PaLM/ST-MoE)

    # ── Flagship Stabilization (Gemma 2 / DeepSeek-V3 style) ──────────────
    # Softcapping bounds final logits to prevent confidence explosion (vanishing gradients)
    final_logit_softcap: float = 30.0     # 0.0 disables softcapping
    # Post-norm on embeddings provides a stable, unit-variance input manifold
    embed_post_norm: bool = True

    # ── Multi-Token Prediction (MTP) ──────────────────────────────────────
    num_mtp_heads: int = 2                # predict t+2, t+3 etc. (0 for standard LM)
    mtp_loss_coef: float = 0.5            # weighting for the averaged MTP loss

    # ── Experimental "Smartness" Enhancements ─────────────────────────────
    # Latent Pondering: number of recurrent loops before the LM head.
    ponder_steps: int = 0                 
    # Subtractive MoE: dedicate Expert 0 to mathematically suppress activations
    use_critic_expert: bool = False
    # Cross-Layer Workspace: global parallel memory stream shared across layers
    use_memory_workspace: bool = False

    # ── Mixed Precision ───────────────────────────────────────────────────
    use_fp8: bool = False                 # Whether to use FP8 precision where supported

    # ── Training ──────────────────────────────────────────────────────
    # (used by train scripts, not the model itself)
    block_size: int = 2048            # context window used during training
    # Gradient checkpointing: recompute activations during backward to save VRAM.
    # Set True when VRAM is the bottleneck (adds ~30% compute overhead).
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        # Derived: head_dim
        assert self.hidden_dim % self.n_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
        )
        self.head_dim = self.hidden_dim // self.n_heads

        # Derived: ffn_dim — rounded to nearest multiple of 64
        raw = int(8 / 3 * self.hidden_dim)
        self.ffn_dim = math.ceil(raw / 64) * 64

        # Sanity: GQA constraint
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )

        # Vocabulary: pad to multiple of 64 for matmul efficiency
        remainder = self.vocab_size % self.pad_to_multiple
        if remainder != 0:
            self.vocab_size = self.vocab_size + (self.pad_to_multiple - remainder)

    @property
    def kv_groups(self) -> int:
        """Number of query heads sharing each KV head."""
        return self.n_heads // self.n_kv_heads

    def param_count_estimate(self) -> int:
        """Rough parameter count estimate (embeddings + layers)."""
        embed = self.vocab_size * self.hidden_dim
        per_layer = (
            # QKV projections
            self.hidden_dim * self.hidden_dim                      # Q
            + self.hidden_dim * self.n_kv_heads * self.head_dim * 2  # KV
            + self.hidden_dim * self.hidden_dim                    # O
            # SwiGLU: gate + up + down
            + self.hidden_dim * self.ffn_dim * 2                   # gate + up
            + self.ffn_dim * self.hidden_dim                       # down
            # RMSNorm: 2 per layer (pre-attn, pre-ffn)
            + self.hidden_dim * 2
        )
        final_norm = self.hidden_dim
        lm_head = self.vocab_size * self.hidden_dim                # tied with embed
        return embed + self.n_layers * per_layer + final_norm      # lm_head is tied


# ── Default 300M config ───────────────────────────────────────────────────────
# hidden=1024, layers=24, heads=16, kv_heads=4
# Estimated params: ~295M (with tied embeddings)
DEFAULT_300M = ModelConfig(
    vocab_size=32_768,
    hidden_dim=1024,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=2048,
    rope_theta=10_000.0,
    rope_scaling_factor=1.0,
    rope_scaling_type="yarn",
    dropout=0.0,
    init_std=0.02,
    scale_residual_init=True,
    attn_layer_period=4,
)

# ── 700 Million Parameter Config ────────────────────────────────────────────────
# Requires ~10-12GB VRAM (Perfect for a single Kaggle T4 or P100)
# hidden=1536, layers=24, heads=12, kv_heads=4
FLAGSHIP_700M = ModelConfig(
    vocab_size=32_768,
    hidden_dim=1536,
    n_layers=24,
    n_heads=12,
    n_kv_heads=4,
    max_seq_len=4096,
    rope_theta=10_000.0,
    rope_scaling_factor=1.0,
    rope_scaling_type="yarn",
    dropout=0.0,
    init_std=0.018,
    scale_residual_init=True,
    attn_layer_period=4,
)

# ── 1 Billion Parameter Config ────────────────────────────────────────────────
# Requires ~16GB VRAM (e.g. Google Colab T4, Kaggle P100)
# hidden=2048, layers=24, heads=16, kv_heads=8
FLAGSHIP_1B = ModelConfig(
    vocab_size=32_768,
    hidden_dim=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=8,
    max_seq_len=4096,
    rope_theta=10_000.0,
    rope_scaling_factor=1.0,
    rope_scaling_type="yarn",
    dropout=0.0,
    init_std=0.015,
    scale_residual_init=True,
    attn_layer_period=4,
)

# ── 3 Billion Parameter Config ────────────────────────────────────────────────
# Requires ~32GB-40GB VRAM (e.g. Kaggle T4x2, Colab A100/L4)
# hidden=3072, layers=32, heads=24, kv_heads=8
FLAGSHIP_3B = ModelConfig(
    vocab_size=32_768,
    hidden_dim=3072,
    n_layers=32,
    n_heads=24,
    n_kv_heads=8,
    max_seq_len=4096,
    rope_theta=10_000.0,
    rope_scaling_factor=1.0,
    rope_scaling_type="yarn",
    dropout=0.0,
    init_std=0.01,
    scale_residual_init=True,
    attn_layer_period=4,
)

# ── Mac-optimized "Nano" config (~20M params) ─────────────────────────────────
# Designed for training on Apple Silicon (M1/M2/M3) at real-time speeds.
# 512 context, 8 layers, 512 hidden dim — big enough to learn language,
# fast enough to see progress in minutes instead of hours per step.
MAC_NANO = ModelConfig(
    vocab_size=32_768,
    hidden_dim=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=512,
    rope_theta=10_000.0,
    rope_scaling_factor=1.0,
    rope_scaling_type="yarn",
    dropout=0.0,
    init_std=0.02,
    scale_residual_init=True,
    attn_layer_period=4,
    block_size=512,
)

# ── Tiny config for fast unit tests ──────────────────────────────────────────
TINY_TEST = ModelConfig(
    vocab_size=256,
    hidden_dim=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=128,
    rope_theta=10_000.0,
    dropout=0.0,
    init_std=0.02,
    scale_residual_init=True,
    block_size=128,           # must equal max_seq_len for tiny
    attn_layer_period=4,
)
