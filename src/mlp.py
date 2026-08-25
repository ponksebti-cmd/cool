"""
mlp.py — SwiGLU Feed-Forward Network.

SwiGLU (Shazeer, 2020) replaces the standard ReLU FFN with a gated
variant that uses two parallel projections (gate + up) and a SiLU
activation gate. Empirically it outperforms GeLU/ReLU with the same
parameter count when the FFN width is adjusted accordingly.

    FFN(x) = (SiLU(W_gate(x)) ⊙ W_up(x)) @ W_down

Where ⊙ is elementwise multiplication.

Note: no bias terms, following LLaMA convention.
      ffn_dim is derived in ModelConfig to match total param budget.
"""

from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward block.

    Parameters
    ----------
    config : ModelConfig
    layer_idx : int
        Stored for use during scaled residual initialization in model.py.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Gate and up projections: both map hidden_dim -> ffn_dim
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj   = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)

        # Down projection maps ffn_dim -> hidden_dim
        # This is the residual branch; its init is scaled in model.py.
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        # SiLU(gate) ⊙ up — fused via F.silu
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
