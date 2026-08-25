"""
attention.py — Grouped-Query Attention (GQA) with RoPE and QK-Norm.

Architecture decisions:
  - GQA: n_kv_heads < n_heads, KV tensors are expanded (repeat_kv)
    before the scaled dot product. This keeps the memory footprint
    proportional to n_kv_heads, not n_heads.
  - Pre-norm is handled by the parent TransformerBlock; this module
    receives already-normed input.
  - Causal masking: for standard auto-regressive decoding we use
    PyTorch's built-in is_causal=True in scaled_dot_product_attention.
    For packed sequences, an explicit block-diagonal mask is passed in
    (attn_mask parameter), which overrides the causal flag.
  - QK-Norm (Dehghani et al. / DeepSeek-V3 / Gemma 2):
    RMSNorm is applied to Q and K after projection, before the dot
    product. This prevents attention logit explosion in deep networks
    and eliminates attention entropy collapse ("attention sinks").
    Cost: 2 × n_heads × head_dim extra learnable scalars (~8K params).
  - Scaled residual init: o_proj weight is divided by sqrt(2 * n_layers)
    after default init (see model.py _init_weights).
  - No bias on QKV or O projections (following LLaMA/Mistral convention).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope import RotaryEmbedding
from .config import ModelConfig


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """
    Expand KV heads to match Q heads for GQA.

    Input:  [batch, n_kv_heads, seq_len, head_dim]
    Output: [batch, n_heads, seq_len, head_dim]

    We use expand + reshape instead of repeat to avoid copying data.
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class QKNorm(nn.Module):
    """
    Per-head RMSNorm applied to Q and K projections before the dot product.

    Used in Gemma 2, DeepSeek-V3, Chameleon (Meta), and Griffin (DeepMind).
    Prevents attention logit explosion at depth and eliminates attention sinks.

    Each head gets its own independent learnable scale, allowing the model
    to learn different calibration per head.

    Args:
        n_heads  : number of heads being normalized
        head_dim : dimension of each head
        eps      : numerical stability epsilon
    """

    def __init__(self, n_heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Independent scale per head, per dimension
        self.weight = nn.Parameter(torch.ones(n_heads, 1, head_dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, n_heads, seq_len, head_dim]
        Returns:
            normalized x with same shape
        """
        # Normalize in float32 for stability, cast back
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        normed = (x_f32 * rms).to(x.dtype)
        # weight broadcasts: [n_heads, 1, head_dim] over [B, n_heads, T, head_dim]
        return normed * self.weight


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention with RoPE and QK-Norm.

    Parameters
    ----------
    config : ModelConfig
        Provides hidden_dim, n_heads, n_kv_heads, head_dim, dropout, rope_theta.
    layer_idx : int
        Used for scaled residual initialization in model.py.
    rope : RotaryEmbedding
        Shared instance across all layers (weights-free, just a cache).
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_groups = config.kv_groups
        self.layer_idx = layer_idx
        self.dropout = config.dropout
        self.rope = rope

        self.scale = self.head_dim ** -0.5

        # ── Projections (no bias, following LLaMA) ────────────────────────
        self.q_proj = nn.Linear(config.hidden_dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.hidden_dim, bias=False)

        # ── QK-Norm (Gemma 2 / DeepSeek-V3 style) ────────────────────────
        # Applied after projection and reshape, before RoPE.
        # Prevents logit explosion and attention entropy collapse at depth.
        self.q_norm = QKNorm(config.n_heads, config.head_dim)
        self.k_norm = QKNorm(config.n_kv_heads, config.head_dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        is_causal: bool = True,
    ) -> Tensor:
        """
        Args:
            x          : [batch, seq_len, hidden_dim]
            attn_mask  : Optional additive mask [batch, 1, seq_len, seq_len]
                         or boolean mask [batch, seq_len, seq_len].
                         If provided, is_causal is set to False internally.
            position_ids: [batch, seq_len] — for packed sequences with
                         non-contiguous position indices.
            is_causal  : Use PyTorch's fast causal path when no explicit mask.

        Returns:
            [batch, seq_len, hidden_dim]
        """
        B, T, _ = x.shape

        # ── Project ───────────────────────────────────────────────────────
        q = self.q_proj(x)  # [B, T, n_heads * head_dim]
        k = self.k_proj(x)  # [B, T, n_kv_heads * head_dim]
        v = self.v_proj(x)  # [B, T, n_kv_heads * head_dim]

        # ── Reshape to [B, heads, T, head_dim] ───────────────────────────
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ── QK-Norm (applied before RoPE) ─────────────────────────────────
        # Normalizing before RoPE ensures the rotational encoding is applied
        # to unit-scale queries/keys, preventing logit blow-up at depth.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ── Apply RoPE ────────────────────────────────────────────────────
        q, k = self.rope(q, k, position_ids=position_ids)

        # ── Expand KV heads for GQA ───────────────────────────────────────
        k = repeat_kv(k, self.kv_groups)
        v = repeat_kv(v, self.kv_groups)

        # ── Scaled dot-product attention ──────────────────────────────────
        # When an explicit mask is provided (packed sequences), disable the
        # is_causal shortcut because the mask already encodes causality.
        effective_causal = is_causal and (attn_mask is None)
        dropout_p = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=effective_causal,
            scale=self.scale,
        )

        # ── Merge heads and project ───────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        return self.o_proj(out)
