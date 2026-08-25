"""
linear.py — Gated DeltaNet (Linear Attention) with data-dependent output gate.

Provides a pure PyTorch formulation of a Gated Linear Attention style block.
Instead of relying on external C++ kernels, it leverages cumulative sums for
a parallel scan, which is efficient for our context lengths.

Architecture (upgraded from baseline):
  - Standard GatedDeltaNet: Q @ (K^T * decay_D) @ V
  - Data-dependent output gate (Hawk/Griffin, DeepMind 2024; GLA, Yang et al. 2024):
      gate = sigmoid(g_proj(x))     # [B, T, hidden_dim]
      out  = (attention_output * gate)
    This selective gate dramatically improves the block's ability to suppress
    irrelevant state information, bringing linear attention much closer to
    full attention quality on tasks requiring selective memory.

Mathematically, it computes:
    D[t, j] = exp(cum_log_beta[t] - cum_log_beta[j])  for j <= t
    Output = sigmoid(g(x)) * ((Q @ K^T * D) @ V)

This is an exact O(T^2) reference implementation of the recurrent O(T) state update:
    S_t = beta_t * S_{t-1} + K_t * V_t^T
    O_t = sigmoid(g_t) * Q_t * S_t

It uses no RoPE because the exponential decay beta natively models relative positions.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet with data-dependent output gate.
    Acts as a drop-in replacement for GroupedQueryAttention.

    The output gate (g_proj) adds a per-position, per-channel gating
    mechanism on top of the linear attention output. At zero cost in
    expressivity, this allows the model to learn "when to trust the
    linear attention state" — analogous to the forget gate in an LSTM.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx

        # Projections (no biases, matching GQA)
        self.q_proj = nn.Linear(config.hidden_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.n_heads * self.head_dim, bias=False)

        # Gating projection for exponential decay (data-dependent beta)
        self.beta_proj = nn.Linear(config.hidden_dim, self.n_heads, bias=False)

        # Data-dependent output gate (Hawk/Griffin / GLA style)
        # Maps hidden_dim -> hidden_dim (same cost as one QKV projection)
        # sigmoid activation applied in forward pass
        self.g_proj = nn.Linear(config.hidden_dim, self.n_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_dim, bias=False)

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
            position_ids: Ignored (decay provides implicit positional encoding).
            is_causal  : Whether to apply a causal mask.
        Returns:
            [batch, seq_len, hidden_dim]
        """
        B, T, _ = x.shape

        # ── Project ───────────────────────────────────────────────────────
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # DeltaNet applies L2 normalization to Q and K for numerical stability
        q = F.normalize(q, p=2.0, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2.0, dim=-1, eps=1e-6)

        # ── Data-dependent output gate ────────────────────────────────────
        # Computed from the original (pre-norm) input x, so it sees the
        # full residual signal before linear attention squashes information.
        # Shape: [B, T, n_heads * head_dim] -> [B, n_heads, T, head_dim]
        gate = torch.sigmoid(self.g_proj(x))
        gate = gate.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ── Data-dependent decay (beta) ───────────────────────────────────
        # beta: [B, H, T, 1] - bounded strictly between 0 and 1
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, self.n_heads, 1).transpose(1, 2)

        # To compute the cumulative product of betas robustly:
        # log(beta) -> cumsum -> exp
        log_beta = torch.log(beta + 1e-6)
        cum_log_beta = torch.cumsum(log_beta, dim=2)  # [B, H, T, 1]

        # Decay matrix D[t, j] = exp(cum_log_beta[t] - cum_log_beta[j])
        log_D = cum_log_beta - cum_log_beta.transpose(-2, -1)  # [B, H, T, T]

        # ── Masking ───────────────────────────────────────────────────────
        # We must apply the mask BEFORE exp() to avoid exp(large_positive) -> inf,
        # which would result in inf * 0.0 = NaN in the backward pass.
        if is_causal or (attn_mask is not None):
            causal_mask = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
            log_D = log_D.masked_fill(~causal_mask, float('-inf'))

        # Apply block-diagonal mask for packed sequences
        if attn_mask is not None:
            log_D = log_D.masked_fill(attn_mask == float('-inf'), float('-inf'))

        D = torch.exp(log_D)

        # ── Linear Attention ──────────────────────────────────────────────
        # scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * D

        # Aggregate V: [B, H, T, head_dim]
        out = torch.matmul(scores, v)

        # ── Apply data-dependent output gate ──────────────────────────────
        # gate selectively amplifies or suppresses each output dimension,
        # allowing the model to decide how much to trust the linear state.
        out = out * gate

        # ── Merge heads and project ───────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        return self.o_proj(out)
