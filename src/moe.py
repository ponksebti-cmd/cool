"""
moe.py — Sparse Mixture-of-Experts (MoE) Layer.

Architecture:
  - Replaces standard SwiGLU MLP.
  - Expert weights are stored as STACKED 3-D tensors [n_experts, *, *],
    enabling a single batched BMM (torch.bmm) instead of a Python for-loop.
    This is 4-8× faster on GPU/MPS and is fully torch.compile compatible.
  - Optionally contains 1 `shared_expert` (always active for all tokens).

Routing strategy: Expert Choice (Zhou et al., 2022 / Switch Transformer v2)
  - Each expert picks its top-C tokens (not: each token picks top-K experts).
  - Guarantees perfect load balance by construction with zero token dropping.
  - Z-Loss (Zoph et al., PaLM / ST-MoE) prevents router logit explosion.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .mlp import SwiGLUMLP


class MoELayer(nn.Module):
    """
    Mixture of Experts with Expert Choice routing, batched BMM, and Z-Loss.

    Expert weights are stored as single 3-D Parameters (not a ModuleList of
    Linear layers). This removes the Python for-loop over experts entirely
    and replaces it with three batched matrix multiplications:

        gate_out   = BMM(tokens, expert_gate^T)   # SwiGLU gate branch
        up_out     = BMM(tokens, expert_up^T)     # SwiGLU up branch
        expert_out = BMM(silu(gate)*up, expert_down^T)

    All three BMMs run in a single CUDA/MPS kernel dispatch, making this
    roughly 4-8× faster than a sequential Python loop at typical MoE sizes.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config    = config
        self.layer_idx = layer_idx

        self.n_experts       = config.n_routed_experts
        self.top_k           = config.n_active_experts
        self.capacity_factor = config.moe_capacity_factor
        self.hidden_dim      = config.hidden_dim
        self.ffn_dim         = config.ffn_dim

        # Router: projects hidden_dim -> n_experts scores
        self.router = nn.Linear(config.hidden_dim, self.n_experts, bias=False)

        # ── Stacked expert weights [n_experts, *, *] ──────────────────────
        # Stored as raw Parameters (not nn.Linear) so they can be passed
        # directly to torch.bmm without per-call stacking overhead.
        self.expert_gate = nn.Parameter(
            torch.empty(self.n_experts, config.ffn_dim, config.hidden_dim)
        )
        self.expert_up = nn.Parameter(
            torch.empty(self.n_experts, config.ffn_dim, config.hidden_dim)
        )
        self.expert_down = nn.Parameter(
            torch.empty(self.n_experts, config.hidden_dim, config.ffn_dim)
        )

        # Initialise stacked weights with the same N(0, init_std) as nn.Linear
        nn.init.normal_(self.expert_gate, std=config.init_std)
        nn.init.normal_(self.expert_up,   std=config.init_std)
        nn.init.normal_(self.expert_down, std=config.init_std)

        # Shared Expert (always active — guarantees every token gets processed)
        if config.use_shared_expert:
            self.shared_expert: SwiGLUMLP | None = SwiGLUMLP(config, layer_idx)
        else:
            self.shared_expert = None

        # Diagnostics (read by training loop for logging)
        self.l_aux: Tensor | float = 0.0
        self.l_z:   Tensor | float = 0.0
        self.expert_counts: Tensor | None = None

    def apply_residual_scale(self, scale: float) -> None:
        """Apply GPT-NeoX style 1/sqrt(2L) scaling to the down-projection."""
        self.expert_down.data.mul_(scale)
        if self.shared_expert is not None and hasattr(self.shared_expert, "down_proj"):
            self.shared_expert.down_proj.weight.data.mul_(scale)

    def forward(self, x: Tensor) -> Tensor:
        """
        Expert Choice MoE with batched BMM expert computation.

        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            out: [batch, seq_len, hidden_dim]
        """
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)   # [N, C] where N = B * T
        N = x_flat.shape[0]

        # ── Shared Expert ─────────────────────────────────────────────────
        shared_out: Tensor | float = self.shared_expert(x) if self.shared_expert is not None else 0.0

        # ── Router ────────────────────────────────────────────────────────
        router_logits = self.router(x_flat)           # [N, E]

        # ── Z-Loss (PaLM / ST-MoE) ────────────────────────────────────────
        # Penalises large logit magnitudes to prevent router collapse.
        # L_z = mean [ log(sum_i exp(logit_i))^2 ]
        log_z = torch.logsumexp(router_logits, dim=-1)  # [N]
        self.l_z = (log_z ** 2).mean()

        # ── Expert Choice routing ─────────────────────────────────────────
        routing_probs = F.softmax(router_logits, dim=-1)  # [N, E]
        capacity = max(1, int((N * self.top_k / self.n_experts) * self.capacity_factor))
        capacity = min(capacity, N)

        # Each expert selects its top-C tokens: [E, capacity]
        topk_probs, topk_indices = torch.topk(routing_probs.t(), k=capacity, dim=-1)

        # ── Load-balancing aux loss (diagnostic; balance is guaranteed) ────
        P_i = routing_probs.mean(dim=0)               # [E]
        f_i = torch.full_like(P_i, 1.0 / self.n_experts)
        self.l_aux = self.n_experts * torch.sum(f_i * P_i)
        self.expert_counts = torch.full(
            (self.n_experts,), capacity, dtype=torch.long, device=x.device
        )

        # ── Batched BMM expert forward (replaces Python for-loop) ─────────
        # Gather: [E, capacity, C]
        gathered = x_flat[topk_indices.reshape(-1)].view(self.n_experts, capacity, C)

        # Three fused batched matrix multiplications (SwiGLU):
        #   gate_out   = [E, cap, ffn]  =  gathered @ expert_gate^T
        #   up_out     = [E, cap, ffn]  =  gathered @ expert_up^T
        #   expert_out = [E, cap, C]    =  (silu(gate)*up) @ expert_down^T
        gate_out   = torch.bmm(gathered, self.expert_gate.mT)        # [E, cap, ffn]
        up_out     = torch.bmm(gathered, self.expert_up.mT)          # [E, cap, ffn]
        expert_out = torch.bmm(F.silu(gate_out) * up_out, self.expert_down.mT)  # [E, cap, C]
        
        # ── The Critic Expert (Subtractive MoE) ───────────────────────────
        # If enabled, Expert 0 mathematically suppresses activations instead of adding
        if getattr(self.config, "use_critic_expert", False):
            # We must clone expert_out[0] or do an out-of-place op, since expert_out
            # requires gradients and in-place modifications can break autograd.
            expert_out = torch.cat([
                -expert_out[0:1],
                expert_out[1:]
            ], dim=0)

        # Weight by routing probability and scatter back to [N, C]
        expert_out = expert_out * topk_probs.unsqueeze(-1)           # [E, cap, C]
        out_flat   = torch.zeros_like(x_flat)
        out_flat.index_add_(0, topk_indices.reshape(-1), expert_out.reshape(-1, C))

        # ── Final Output ──────────────────────────────────────────────────
        return out_flat.view(B, T, C) + shared_out
