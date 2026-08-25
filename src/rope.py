"""
rope.py — Rotary Position Embeddings (RoPE) with YaRN scaling.

Implementation follows the original Su et al. (2023) paper, extended with
YaRN (Peng et al., 2023 — "YaRN: Efficient Context Window Extension of Large
Language Models") for non-uniform frequency interpolation.

Key design decisions:
  - Cache is registered as a buffer (not a parameter) so it moves
    with .to(device) / .to(dtype) calls automatically.
  - We store half the head_dim (cos/sin are symmetric), then apply
    the standard interleaved rotation in-place.
  - For GQA: the same function works for both Q (n_heads) and K
    (n_kv_heads) since we rotate each head independently.
  - YaRN scaling (rope_scaling_factor > 1): applies a non-uniform
    interpolation where high-frequency dimensions (local syntax) are
    left untouched and low-frequency dimensions (long-range discourse)
    are linearly interpolated. This is strictly better than naive linear
    interpolation at zero extra parameter cost.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor


# ── YaRN helpers ──────────────────────────────────────────────────────────────

def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base_theta: float,
    max_seq_len: int,
) -> float:
    """
    Compute the RoPE dimension index at which the frequency matches
    `num_rotations` full rotations over `max_seq_len` positions.

    This is the inverse of the standard RoPE frequency formula:
        freq_i = 1 / (base_theta ^ (2i / dim))
        num_rotations = freq_i * max_seq_len / (2 * pi)
    Solving for i gives:
        i = dim / 2 * log(max_seq_len / (2*pi*num_rotations)) / log(base_theta)
    """
    return (dim * math.log(max_seq_len / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base_theta)
    )


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> Tensor:
    """
    Smooth ramp function in [0, 1] over `dim` values from min_val to max_val.
    Used to blend between interpolated and extrapolated RoPE frequencies.
    """
    if min_val == max_val:
        max_val += 0.001  # prevent division by zero
    linear = torch.arange(dim, dtype=torch.float32)
    ramp = (linear - min_val) / (max_val - min_val)
    return ramp.clamp(0.0, 1.0)


def _build_yarn_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    scaling_factor: float,
    device: torch.device,
    dtype: torch.dtype,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """
    Build YaRN-scaled cos/sin tables of shape [seq_len, head_dim // 2].

    YaRN splits frequencies into three regions:
      - High-freq (i < d_low):  NOT interpolated — local position info is preserved.
      - Mid-freq  (d_low <= i < d_high): blended — smooth transition.
      - Low-freq  (i >= d_high): fully linearly interpolated by 1/scaling_factor.

    When scaling_factor == 1.0, the output is identical to standard RoPE.
    """
    half = head_dim // 2

    # Standard per-dimension inverse frequencies: [half]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    if scaling_factor != 1.0:
        # Find the dimension boundaries between the three regions
        # beta_fast / beta_slow are "num_rotations" thresholds (from the YaRN paper)
        d_low  = _yarn_find_correction_dim(beta_fast, head_dim, theta, seq_len)
        d_high = _yarn_find_correction_dim(beta_slow, head_dim, theta, seq_len)

        # Ramp mask: 0 = high-freq (no interpolation), 1 = low-freq (full interpolation)
        ramp = _yarn_linear_ramp_mask(d_low, d_high, half).to(device)

        # Interpolated frequencies: scale down by 1/s
        inv_freq_interp = inv_freq / scaling_factor

        # Blend: high-freq keeps original, low-freq uses interpolated
        inv_freq = inv_freq * (1 - ramp) + inv_freq_interp * ramp

    # Position indices: [seq_len]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Apply magnitude scaling (YaRN uses mscale to compensate for attention entropy change)
    # With mscale=1.0 (our default), this is a no-op.
    freqs = torch.outer(t, inv_freq) * mscale

    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin  # each [seq_len, half]


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """
    Standard RoPE cache (no scaling). Equivalent to YaRN with factor=1.0.
    Kept as a clean fallback.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin  # each [seq_len, half]


# ── Rotation application ──────────────────────────────────────────────────────

def rotate_half(x: Tensor) -> Tensor:
    """
    Rotate pairs of elements: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3].

    Input shape: [..., head_dim]
    Output shape: [..., head_dim]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # first half
    x2 = x[..., half:]   # second half
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q: [batch, n_heads, seq_len, head_dim]
        k: [batch, n_kv_heads, seq_len, head_dim]
        cos: [max_seq_len, head_dim // 2]  — full cache
        sin: [max_seq_len, head_dim // 2]  — full cache
        position_ids: [batch, seq_len] — optional; if None, assumed 0..seq_len-1

    Returns:
        q_rot, k_rot — same shapes as inputs.
    """
    seq_len = q.shape[2]

    if position_ids is None:
        cos_seq = cos[:seq_len]  # [seq_len, half]
        sin_seq = sin[:seq_len]
    else:
        # Gather rows for each position in the batch
        # position_ids: [batch, seq_len]
        cos_seq = cos[position_ids]  # [batch, seq_len, half]
        sin_seq = sin[position_ids]

    # Expand cos/sin to broadcast over heads
    # cos_seq: [1, 1, seq_len, half] or [batch, 1, seq_len, half]
    if cos_seq.dim() == 2:
        cos_seq = cos_seq.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half]
        sin_seq = sin_seq.unsqueeze(0).unsqueeze(0)
    else:
        cos_seq = cos_seq.unsqueeze(1)  # [batch, 1, seq_len, half]
        sin_seq = sin_seq.unsqueeze(1)

    # Replicate cos/sin to full head_dim
    # [1, 1, seq_len, half] -> [1, 1, seq_len, head_dim]
    cos_full = torch.cat([cos_seq, cos_seq], dim=-1)
    sin_full = torch.cat([sin_seq, sin_seq], dim=-1)

    q_rot = q * cos_full + rotate_half(q) * sin_full
    k_rot = k * cos_full + rotate_half(k) * sin_full
    return q_rot, k_rot


# ── Module ────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Module wrapper that pre-caches the RoPE cos/sin tables.

    Supports YaRN scaling (Peng et al., 2023) via rope_scaling_factor.
    When rope_scaling_factor == 1.0, behavior is identical to standard RoPE.

    Registered as buffer so device/dtype transfers are automatic.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "yarn",
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type

        cos, sin = self._build_cache(torch.device("cpu"), torch.float32)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def _build_cache(self, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Build the appropriate cache based on scaling settings."""
        if self.scaling_factor == 1.0 or self.scaling_type == "linear":
            # Standard or naive linear interpolation
            if self.scaling_factor != 1.0:
                # Linear interpolation: just scale theta
                theta_scaled = self.theta * self.scaling_factor
            else:
                theta_scaled = self.theta
            return _build_rope_cache(
                self.max_seq_len, self.head_dim, theta_scaled, device, dtype
            )
        else:
            # YaRN: non-uniform interpolation (Peng et al., 2023)
            return _build_yarn_rope_cache(
                self.max_seq_len,
                self.head_dim,
                self.theta,
                self.scaling_factor,
                device,
                dtype,
            )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # Cast cache to match q dtype (handles bf16/fp16 automatically)
        cos = self.cos_cache.to(q.dtype)
        sin = self.sin_cache.to(q.dtype)
        return apply_rope(q, k, cos, sin, position_ids)
