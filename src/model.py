"""
model.py — Decoder-only Transformer (Stage 1 baseline).

Architecture: standard dense transformer with:
  - RMSNorm (pre-norm on both attention and MLP sub-blocks)
  - Grouped-Query Attention (GQA) with RoPE
  - SwiGLU MLP
  - Shared RotaryEmbedding across layers (no per-layer overhead)
  - Scaled residual initialization (Wang et al., DeepNet)
  - Tied input/output embeddings (saves ~32M params at 300M scale)

Initialization strategy (Wang et al., 2022 / Sho Takase et al.):
  - Standard linear layers: N(0, σ) where σ = config.init_std = 0.02
  - Residual projections (o_proj, down_proj): N(0, σ / sqrt(2 * n_layers))
    This prevents variance accumulation across depth and has been shown to
    enable stable training without LR warmup in very deep networks.

The model is designed to be forward-compatible with:
  - Stage 2: hybrid attention (swap some TransformerBlock.attn with linear recurrent)
  - Stage 3: MoE (swap TransformerBlock.mlp with MoELayer)
  - Stage 4: MTP heads (add extra prediction heads at the top)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .rope import RotaryEmbedding
from .attention import GroupedQueryAttention
from .mlp import SwiGLUMLP
from .linear import GatedDeltaNet
from .moe import MoELayer


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does NOT subtract the mean — it only
    normalizes by the RMS. This removes one operation and has been
    shown to work equally well in practice (LLaMA, Mistral, etc.).

    x_norm = x / rms(x) * weight    where rms(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 for numerical stability, cast back afterward
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        normed = x_f32 * rms
        return normed.to(x.dtype) * self.weight


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single transformer decoder layer with pre-norm.

    Pre-norm layout (as used in LLaMA/Mistral/GPT-NeoX):
        x = x + Attn(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    This is more training-stable than post-norm at large scale.

    The attn and mlp attributes are designed as swappable slots so that
    later stages can replace them without touching this class.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rope: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm for attention sub-block
        self.norm_attn = RMSNorm(config.hidden_dim)
        # Pre-norm for MLP sub-block
        self.norm_mlp = RMSNorm(config.hidden_dim)

        # Swappable slots: Hybrid Attention
        # Pattern: 1 dense GQA out of every `attn_layer_period` layers
        if (layer_idx + 1) % config.attn_layer_period == 0:
            self.attn: nn.Module = GroupedQueryAttention(config, layer_idx, rope)
        else:
            self.attn: nn.Module = GatedDeltaNet(config, layer_idx)

        # Swappable slots (Stage 3 replaces mlp)
        self.mlp: nn.Module = MoELayer(config, layer_idx)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x           : [batch, seq_len, hidden_dim]
            attn_mask   : Optional [batch, 1, seq_len, seq_len] additive mask.
            position_ids: Optional [batch, seq_len] for packed sequences.
        Returns:
            [batch, seq_len, hidden_dim]
        """
        # Attention residual stream
        x = x + self.attn(
            self.norm_attn(x),
            attn_mask=attn_mask,
            position_ids=position_ids,
            is_causal=(attn_mask is None),
        )
        # MLP residual stream
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ── Full Model ────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Decoder-only Transformer Language Model.

    Tied embeddings: lm_head.weight is shared with tok_emb.weight,
    reducing parameters by ~33M at 300M scale.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding table
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Embedding post-norm (DeepSeek-V3 / Gemma 2)
        if getattr(config, "embed_post_norm", True):
            self.norm_emb = RMSNorm(config.hidden_dim)
        else:
            self.norm_emb = None

        # Shared RoPE cache (lives on the same device as the model)
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
            scaling_type=getattr(config, "rope_scaling_type", "yarn"),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i, rope=self.rope)
            for i in range(config.n_layers)
        ])

        # Final normalization before the LM head
        self.norm_out = RMSNorm(config.hidden_dim)

        # Language model head (output projection)
        # Weight is tied to tok_emb: lm_head shares the same Parameter object.
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying

        # Multi-Token Prediction (MTP) Heads
        self.num_mtp_heads = getattr(config, "num_mtp_heads", 0)
        if self.num_mtp_heads > 0:
            self.mtp_blocks = nn.ModuleList([
                TransformerBlock(config, layer_idx=config.n_layers + i, rope=self.rope)
                for i in range(self.num_mtp_heads)
            ])
            self.mtp_norms = nn.ModuleList([
                RMSNorm(config.hidden_dim) for _ in range(self.num_mtp_heads)
            ])
            
        # ── Experimental Smartness ────────────────────────────────────────
        self.ponder_steps = getattr(config, "ponder_steps", 0)
        if self.ponder_steps > 0:
            self.ponder_block = nn.Sequential(
                RMSNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim * 4, bias=False),
                nn.SiLU(),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim, bias=False)
            )
            nn.init.normal_(self.ponder_block[1].weight, std=config.init_std)
            nn.init.normal_(self.ponder_block[3].weight, std=config.init_std / math.sqrt(2 * max(1, self.ponder_steps)))
            
        self.use_workspace = getattr(config, "use_memory_workspace", False)
        if self.use_workspace:
            self.workspace_norm = RMSNorm(config.hidden_dim)
            self.workspace_write = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            nn.init.normal_(self.workspace_write.weight, std=config.init_std / math.sqrt(2 * config.n_layers))
        
        if self.num_mtp_heads > 0:
            # Medusa-style independent projection MLPs per future token
            # Final projection is tied to lm_head.weight
            self.mtp_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim, bias=False),
                    nn.SiLU()
                ) for _ in range(self.num_mtp_heads)
            ])

        # ── Initialize weights ──────────────────────────────────────────
        self.apply(self._init_weights)
        if config.scale_residual_init:
            self._apply_residual_scaling()

    # ── Initialization ────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """Standard init: N(0, 0.02) for linear & embedding weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _apply_residual_scaling(self) -> None:
        """
        Scale residual projection weights by 1/sqrt(2 * n_layers).

        This is applied after _init_weights so the effective std becomes:
        Applies GPT-NeoX style scaling (1/sqrt(2L)) to residual projections.
        """
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for layer in self.layers:
            # Scale attention output projection
            if hasattr(layer.attn, "o_proj"):
                layer.attn.o_proj.weight.data.mul_(scale)

            # Scale MLP/MoE down projections
            # MoELayer exposes apply_residual_scale() since weights are stacked 3-D tensors
            if hasattr(layer.mlp, "apply_residual_scale"):
                layer.mlp.apply_residual_scale(scale)
            elif hasattr(layer.mlp, "down_proj"):
                layer.mlp.down_proj.weight.data.mul_(scale)

    # ── Parameter count ───────────────────────────────────────────────────

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count trainable parameters, optionally excluding embedding table."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            total -= self.tok_emb.weight.numel()
        return total

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            input_ids   : [batch, seq_len]  — token IDs
            attn_mask   : Optional additive attention mask for packed sequences.
                          Shape: [batch, 1, seq_len, seq_len] (float, 0/-inf values)
            position_ids: Optional [batch, seq_len] — non-contiguous positions
                          for packed sequences.
            targets     : Optional [batch, seq_len] — labels for loss computation.
                          If provided, returns (logits, loss). Loss is computed
                          with cross-entropy ignoring index -100.

        Returns:
            (logits, loss)
            logits: [batch, seq_len, vocab_size]
            loss  : scalar Tensor or None
        """
        # ── Embed ─────────────────────────────────────────────────────────
        x = self.tok_emb(input_ids)           # [B, T, D]
        if self.norm_emb is not None:
            x = self.norm_emb(x)

        # ── Transformer layers ────────────────────────────────────────────
        # Gradient checkpointing trades compute for memory: activations are
        # NOT stored during the forward pass; instead they are recomputed
        # during the backward pass. Controlled by config.use_gradient_checkpointing.
        use_ckpt = getattr(self.config, "use_gradient_checkpointing", False) and self.training
        
        workspace = torch.zeros_like(x) if self.use_workspace else None

        for layer in self.layers:
            # 1. Read from parallel workspace stream
            if workspace is not None:
                x = x + self.workspace_norm(workspace)
                
            if use_ckpt:
                # use_reentrant=False is recommended in PyTorch >= 2.0 for
                # better compatibility with torch.compile and autocast.
                x = torch.utils.checkpoint.checkpoint(
                    layer, x,
                    attn_mask, position_ids,
                    use_reentrant=False,
                )
            else:
                x = layer(x, attn_mask=attn_mask, position_ids=position_ids)
                
            # 2. Write to parallel workspace stream
            if workspace is not None:
                workspace = workspace + self.workspace_write(x)
                
        # ── Experimental: Latent Pondering ────────────────────────────────
        if self.ponder_steps > 0:
            for _ in range(self.ponder_steps):
                x = x + self.ponder_block(x)

        # ── Final norm ────────────────────────────────────────────────────
        x = self.norm_out(x)

        # ── LM head ───────────────────────────────────────────────────────
        # During training: compute logits for ALL positions (needed for loss).
        # During inference: only the last position matters — callers can slice.
        logits = self.lm_head(x)              # [B, T, V]
        
        # Output Logit Softcapping (Gemma 2 / DeepSeek-V3 style)
        # Bounding the logits strictly prevents extreme overconfidence and
        # vanishing gradients during deep stages of training.
        softcap = getattr(self.config, "final_logit_softcap", 0.0)
        if softcap > 0.0:
            # We explicitly divide, tanh, and multiply.
            logits = softcap * torch.tanh(logits / softcap)

        # ── Loss ──────────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            # targets can be [B, T] (standard) or [B, T, K] (MTP)
            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)  # [B, T, 1]
                
            # Base language modeling loss (predicting t+1)
            target_0 = targets[:, :, 0].reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_0,
                ignore_index=-100,
            )
            
            # Multi-Token Prediction (MTP) losses
            if self.num_mtp_heads > 0 and targets.size(-1) > 1:
                mtp_loss_sum = 0.0
                # Process up to min(num_mtp_heads, available_targets-1)
                valid_k = min(self.num_mtp_heads, targets.size(-1) - 1)
                
                for k in range(valid_k):
                    # Project hidden states for k-th future token
                    mtp_h = self.mtp_projections[k](x)
                    # Compute logits (tied to lm_head.weight)
                    mtp_logits = torch.nn.functional.linear(mtp_h, self.lm_head.weight)
                    
                    # Apply softcapping to MTP heads as well
                    if softcap > 0.0:
                        mtp_logits = softcap * torch.tanh(mtp_logits / softcap)

                    # Compute cross entropy for shift k+1
                    step_loss = torch.nn.functional.cross_entropy(
                        mtp_logits.view(-1, mtp_logits.size(-1)),
                        target_k,
                        ignore_index=-100,
                    )
                    mtp_loss_sum = mtp_loss_sum + step_loss
                    
                if valid_k > 0:
                    avg_mtp_loss = mtp_loss_sum / valid_k
                    loss = loss + self.mtp_loss_coef * avg_mtp_loss

        return logits, loss

    # ── Inference helper ──────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """
        Simple greedy/top-k sampling for quick inference tests.
        Not optimized (no KV cache) — for validation only.

        Args:
            input_ids     : [1, prompt_len]
            max_new_tokens: number of tokens to generate
            temperature   : softmax temperature
            top_k         : if set, restrict to top-k logits

        Returns:
            [1, prompt_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            ids = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(ids)
            logits = logits[:, -1, :] / temperature  # [1, V]

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
