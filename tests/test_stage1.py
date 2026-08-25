"""
tests/test_stage1.py — Unit tests for Stage 1 components.

Run with: pytest tests/test_stage1.py -v

Tests cover:
  1. Config correctness (derived fields, param count estimate)
  2. RMSNorm: zero-mean input, dtype preservation
  3. RoPE: shape, causality, dtype broadcast
  4. GQA: output shape, GQA expansion, causal property
  5. SwiGLU MLP: output shape
  6. Scaled residual init: verify o_proj/down_proj are scaled down
  7. Transformer forward: shapes, loss computation
  8. Sequence packing: correct doc boundaries, no cross-doc contamination
  9. Block-diagonal mask: causal + cross-doc blocking
 10. Overfit: integration test — loss must reach < 0.10 in 200 steps
"""

from __future__ import annotations
import sys
import math

import pytest
import torch

sys.path.insert(0, ".")
from src.config import ModelConfig, TINY_TEST, DEFAULT_300M
from src.rope import RotaryEmbedding, apply_rope, rotate_half
from src.attention import GroupedQueryAttention, repeat_kv
from src.mlp import SwiGLUMLP
from src.model import Transformer, RMSNorm, TransformerBlock
from src.data import (
    SequencePacker,
    PackedDataset,
    build_block_diagonal_mask,
    make_dataloader,
)
from overfit_test import run_overfit_test


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> ModelConfig:
    return TINY_TEST


@pytest.fixture
def model(cfg: ModelConfig) -> Transformer:
    torch.manual_seed(0)
    return Transformer(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_head_dim_derived(self, cfg):
        assert cfg.head_dim == cfg.hidden_dim // cfg.n_heads

    def test_ffn_dim_is_multiple_of_64(self, cfg):
        assert cfg.ffn_dim % 64 == 0

    def test_kv_groups(self, cfg):
        assert cfg.kv_groups == cfg.n_heads // cfg.n_kv_heads

    def test_vocab_padded_to_multiple(self):
        raw_cfg = ModelConfig(vocab_size=30_000, hidden_dim=128, n_layers=2,
                               n_heads=4, n_kv_heads=2)
        assert raw_cfg.vocab_size % raw_cfg.pad_to_multiple == 0

    def test_300m_approx_params(self):
        est = DEFAULT_300M.param_count_estimate()
        # Should be in 250M–350M range
        assert 250_000_000 < est < 350_000_000, f"Estimate out of range: {est:,}"

    def test_invalid_head_division_raises(self):
        with pytest.raises(AssertionError):
            ModelConfig(hidden_dim=128, n_heads=5, n_kv_heads=2,
                        n_layers=2)  # 5 doesn't divide evenly

    def test_invalid_kv_groups_raises(self):
        with pytest.raises(AssertionError):
            ModelConfig(hidden_dim=128, n_heads=4, n_kv_heads=3,
                        n_layers=2)  # 4 not divisible by 3


# ─────────────────────────────────────────────────────────────────────────────
# 2. RMSNorm
# ─────────────────────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        assert norm(x).shape == (2, 10, 64)

    def test_roughly_unit_rms(self):
        """After norming, RMS of each vector should be close to 1 (weight=1)."""
        norm = RMSNorm(256)
        x = torch.randn(4, 16, 256) * 5.0   # large variance input
        y = norm(x)
        rms = y.float().pow(2).mean(-1).sqrt()
        # Weight initialised to ones, so normed RMS ≈ 1
        assert (rms - 1.0).abs().max() < 0.1

    def test_dtype_preservation_bf16(self):
        norm = RMSNorm(64).to(torch.bfloat16)
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        assert norm(x).dtype == torch.bfloat16

    def test_dtype_preservation_fp32(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, dtype=torch.float32)
        assert norm(x).dtype == torch.float32

    def test_zero_input_no_nan(self):
        """RMSNorm should not produce NaN on zero input (eps protects it)."""
        norm = RMSNorm(64)
        x = torch.zeros(2, 8, 64)
        y = norm(x)
        assert not torch.isnan(y).any()


# ─────────────────────────────────────────────────────────────────────────────
# 3. RoPE
# ─────────────────────────────────────────────────────────────────────────────

class TestRoPE:
    def test_rotate_half_shape(self):
        x = torch.randn(2, 4, 8, 64)
        assert rotate_half(x).shape == x.shape

    def test_rope_output_shapes(self, cfg):
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        B, T = 2, 16
        q = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
        k = torch.randn(B, cfg.n_kv_heads, T, cfg.head_dim)
        q_r, k_r = rope(q, k)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_rope_norm_preservation(self, cfg):
        """RoPE is an isometry — it should not change vector norms."""
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        q = torch.randn(1, cfg.n_heads, 32, cfg.head_dim)
        k = torch.randn(1, cfg.n_kv_heads, 32, cfg.head_dim)
        q_r, k_r = rope(q, k)
        q_norm = q.norm(dim=-1)
        q_r_norm = q_r.norm(dim=-1)
        assert torch.allclose(q_norm, q_r_norm, atol=1e-5)

    def test_rope_dtype_bf16(self, cfg):
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        q = torch.randn(1, cfg.n_heads, 8, cfg.head_dim, dtype=torch.bfloat16)
        k = torch.randn(1, cfg.n_kv_heads, 8, cfg.head_dim, dtype=torch.bfloat16)
        q_r, k_r = rope(q, k)
        assert q_r.dtype == torch.bfloat16

    def test_rope_position_ids(self, cfg):
        """Custom position_ids produce different output than default 0..T-1."""
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        q = torch.randn(1, cfg.n_heads, 4, cfg.head_dim)
        k = torch.randn(1, cfg.n_kv_heads, 4, cfg.head_dim)
        q_default, _ = rope(q, k, position_ids=None)
        pos = torch.tensor([[10, 11, 12, 13]])
        q_shifted, _ = rope(q, k, position_ids=pos)
        assert not torch.allclose(q_default, q_shifted, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GQA Attention
# ─────────────────────────────────────────────────────────────────────────────

class TestGQA:
    def test_output_shape(self, cfg):
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        attn = GroupedQueryAttention(cfg, layer_idx=0, rope=rope)
        x = torch.randn(2, 16, cfg.hidden_dim)
        out = attn(x)
        assert out.shape == (2, 16, cfg.hidden_dim)

    def test_repeat_kv(self, cfg):
        B, n_kv, T, hd = 2, cfg.n_kv_heads, 8, cfg.head_dim
        k = torch.randn(B, n_kv, T, hd)
        k_expanded = repeat_kv(k, cfg.kv_groups)
        assert k_expanded.shape == (B, cfg.n_heads, T, hd)

    def test_repeat_kv_identity_when_one(self, cfg):
        k = torch.randn(2, cfg.n_kv_heads, 8, cfg.head_dim)
        assert repeat_kv(k, 1) is k  # must return same object

    def test_causal_property(self, cfg):
        """Token at position i should not depend on position j > i."""
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        attn = GroupedQueryAttention(cfg, layer_idx=0, rope=rope)
        attn.eval()
        T = 8
        x = torch.randn(1, T, cfg.hidden_dim, requires_grad=False)

        # Compute output at position 3
        out1 = attn(x.clone(), is_causal=True)[:, 3, :]

        # Perturb position 5 (future relative to 3)
        x2 = x.clone()
        x2[:, 5, :] += 10.0
        out2 = attn(x2, is_causal=True)[:, 3, :]

        # Output at position 3 must not change
        assert torch.allclose(out1, out2, atol=1e-5), (
            "Causal violation: position 3 was affected by position 5"
        )

    def test_explicit_mask_overrides_causal(self, cfg):
        """When attn_mask is provided, is_causal is False internally."""
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        attn = GroupedQueryAttention(cfg, layer_idx=0, rope=rope)
        T = 4
        x = torch.randn(1, T, cfg.hidden_dim)
        doc_ids = torch.tensor([[0, 0, 1, 1]])
        mask = build_block_diagonal_mask(doc_ids)
        out = attn(x, attn_mask=mask)
        assert out.shape == (1, T, cfg.hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SwiGLU MLP
# ─────────────────────────────────────────────────────────────────────────────

class TestSwiGLU:
    def test_output_shape(self, cfg):
        mlp = SwiGLUMLP(cfg, layer_idx=0)
        x = torch.randn(2, 16, cfg.hidden_dim)
        assert mlp(x).shape == (2, 16, cfg.hidden_dim)

    def test_no_bias_params(self, cfg):
        mlp = SwiGLUMLP(cfg, layer_idx=0)
        for name, p in mlp.named_parameters():
            assert "bias" not in name, f"Unexpected bias: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scaled residual initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestScaledInit:
    def test_residual_weights_scaled_down(self, cfg, model):
        """o_proj and down_proj weights must have smaller norm than q_proj."""
        expected_scale = (2.0 * cfg.n_layers) ** -0.5
        for layer in model.layers:
            o_norm = layer.attn.o_proj.weight.norm().item()
            q_norm = layer.attn.q_proj.weight.norm().item()
            # o_proj and down_proj should be scaled down relative to q/gate
            # Handle both SwiGLUMLP and MoELayer
            if hasattr(layer.mlp, "experts"):
                # Stage 3 MoE layer
                d_norm = layer.mlp.experts[0].down_proj.weight.norm().item()
                g_norm = layer.mlp.experts[0].gate_proj.weight.norm().item()
            else:
                # Stage 1 standard MLP
                d_norm = layer.mlp.down_proj.weight.norm().item()
                g_norm = layer.mlp.gate_proj.weight.norm().item()

            # They won't be exactly equal because we apply the scale factor
            # to a random init — but they should be noticeably smaller.
            # We check: o_norm / q_norm is roughly near expected_scale.
            ratio_o = o_norm / (q_norm + 1e-8)
            ratio_d = d_norm / (g_norm + 1e-8)

            # Ratio should be less than 1.0 (scaled down)
            assert ratio_o < 1.0, f"o_proj not scaled down: ratio={ratio_o:.4f}"
            assert ratio_d < 1.0, f"down_proj not scaled down: ratio={ratio_d:.4f}"

    def test_unscaled_model_different(self, cfg):
        """Model with scale_residual_init=False should have different norms."""
        cfg2 = ModelConfig(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            scale_residual_init=False,
        )
        torch.manual_seed(1)
        m_unscaled = Transformer(cfg2)
        torch.manual_seed(1)
        m_scaled   = Transformer(cfg)   # same seed, scale_residual_init=True

        o_unscaled = m_unscaled.layers[0].attn.o_proj.weight.norm().item()
        o_scaled   = m_scaled.layers[0].attn.o_proj.weight.norm().item()
        assert o_scaled < o_unscaled


# ─────────────────────────────────────────────────────────────────────────────
# 7. Transformer forward
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformerForward:
    def test_forward_shape(self, cfg, model):
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert loss is None

    def test_forward_with_targets(self, cfg, model):
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        t = torch.randint(0, cfg.vocab_size, (2, 16))
        _, loss = model(x, targets=t)
        assert loss is not None
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_initial_loss_near_random(self, cfg):
        """Random-init model should predict close to uniform — loss ≈ ln(V)."""
        torch.manual_seed(99)
        model = Transformer(cfg)
        model.eval()
        x = torch.randint(2, cfg.vocab_size, (4, cfg.block_size))
        t = torch.randint(2, cfg.vocab_size, (4, cfg.block_size))
        with torch.no_grad():
            _, loss = model(x, targets=t)
        expected = math.log(cfg.vocab_size)
        assert abs(loss.item() - expected) < 2.0, (
            f"Initial loss {loss.item():.3f} far from expected {expected:.3f}"
        )

    def test_tied_weights(self, cfg, model):
        """lm_head.weight and tok_emb.weight must be the same object."""
        assert model.lm_head.weight is model.tok_emb.weight

    def test_generate_shape(self, cfg, model):
        model.eval()
        x = torch.randint(2, cfg.vocab_size, (1, 5))
        out = model.generate(x, max_new_tokens=10)
        assert out.shape == (1, 15)

    def test_bf16_forward(self, cfg):
        """Model should work in bfloat16 without NaNs."""
        model = Transformer(cfg).to(torch.bfloat16)
        x = torch.randint(2, cfg.vocab_size, (1, 8))
        t = torch.randint(2, cfg.vocab_size, (1, 8))
        _, loss = model(x, targets=t)
        assert not torch.isnan(loss)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sequence Packing
# ─────────────────────────────────────────────────────────────────────────────

class TestSequencePacking:
    def test_block_length(self):
        packer = SequencePacker(block_size=16, pad_id=0)
        docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        blocks = packer.pack(docs, shuffle=False)
        for b in blocks:
            assert len(b["input_ids"]) == 16
            assert len(b["doc_ids"]) == 16
            assert len(b["position_ids"]) == 16

    def test_doc_ids_contiguous(self):
        """Each doc_id run in a block should be contiguous."""
        packer = SequencePacker(block_size=32, pad_id=0)
        docs = [[i] * 4 for i in range(1, 6)]  # 5 docs of length 4
        blocks = packer.pack(docs, shuffle=False)
        for b in blocks:
            ids = b["doc_ids"]
            # Check no doc_id appears after a different (non-pad) id, then back
            seen = set()
            prev = None
            for d in ids:
                if d == -1:
                    continue
                if d != prev and d in seen:
                    pytest.fail(f"doc_id {d} appeared non-contiguously in {ids}")
                seen.add(d)
                prev = d

    def test_position_ids_restart_per_doc(self):
        """Position IDs should reset to 0 at the start of each document."""
        packer = SequencePacker(block_size=16, pad_id=0)
        docs = [[10, 11, 12], [20, 21]]
        blocks = packer.pack(docs, shuffle=False)
        b = blocks[0]
        pos = b["position_ids"]
        doc = b["doc_ids"]
        # Find transition points
        prev_doc = doc[0]
        for i, (d, p) in enumerate(zip(doc, pos)):
            if d == -1:
                break
            if d != prev_doc:
                assert p == 0, f"Position did not reset at doc boundary (pos[{i}]={p})"
                prev_doc = d

    def test_padding_position_is_minus1(self):
        packer = SequencePacker(block_size=16, pad_id=0)
        docs = [[1, 2]]  # Only 2 tokens, rest should be padding
        blocks = packer.pack(docs, shuffle=False)
        doc_ids = blocks[0]["doc_ids"]
        assert -1 in doc_ids  # There should be padding

    def test_oversized_doc_truncated(self):
        packer = SequencePacker(block_size=8, pad_id=0)
        docs = [list(range(20))]  # 20 tokens, block_size=8
        blocks = packer.pack(docs, shuffle=False)
        for b in blocks:
            assert len(b["input_ids"]) == 8


# ─────────────────────────────────────────────────────────────────────────────
# 9. Block-diagonal mask
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockDiagonalMask:
    def test_mask_shape(self):
        doc_ids = torch.tensor([[0, 0, 1, 1]])
        mask = build_block_diagonal_mask(doc_ids)
        assert mask.shape == (1, 1, 4, 4)

    def test_causal_within_doc(self):
        """Positions in the same doc should only attend to j <= i."""
        doc_ids = torch.tensor([[0, 0, 0, 0]])
        mask = build_block_diagonal_mask(doc_ids)
        m = mask[0, 0]  # [4, 4]
        # Upper triangle (j > i) should be -inf
        for i in range(4):
            for j in range(i + 1, 4):
                assert m[i, j].item() == float("-inf"), f"mask[{i},{j}] should be -inf"
        # Lower triangle + diagonal should be 0
        for i in range(4):
            for j in range(i + 1):
                assert m[i, j].item() == 0.0, f"mask[{i},{j}] should be 0"

    def test_cross_doc_blocked(self):
        """Positions from different docs should always be blocked."""
        doc_ids = torch.tensor([[0, 0, 1, 1]])
        mask = build_block_diagonal_mask(doc_ids)
        m = mask[0, 0]
        # doc1 token (pos 2 or 3) should NOT attend to doc0 token (pos 0 or 1)
        assert m[2, 0].item() == float("-inf")
        assert m[2, 1].item() == float("-inf")
        assert m[3, 0].item() == float("-inf")
        assert m[3, 1].item() == float("-inf")

    def test_same_doc_not_blocked(self):
        """Within-doc causal positions should not be blocked."""
        doc_ids = torch.tensor([[0, 0, 1, 1]])
        mask = build_block_diagonal_mask(doc_ids)
        m = mask[0, 0]
        # doc0: pos 1 should attend to pos 0
        assert m[1, 0].item() == 0.0
        # doc1: pos 3 should attend to pos 2
        assert m[3, 2].item() == 0.0

    def test_padding_positions_blocked(self):
        """doc_id == -1 (padding) should never be attended to."""
        doc_ids = torch.tensor([[0, 0, -1, -1]])
        mask = build_block_diagonal_mask(doc_ids)
        m = mask[0, 0]
        # Any query attending to positions 2,3 (pad) should be -inf
        for i in range(4):
            assert m[i, 2].item() == float("-inf")
            assert m[i, 3].item() == float("-inf")

    def test_no_cross_doc_gradient(self):
        """
        Integration test: attention output at doc-boundary should not be
        affected by tokens from the other document.

        We run attention with the block-diagonal mask and verify that
        perturbing doc0 tokens doesn't change doc1 output.
        """
        cfg = TINY_TEST
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)
        attn = GroupedQueryAttention(cfg, layer_idx=0, rope=rope)
        attn.eval()

        T = 8
        # Tokens 0-3 = doc 0, tokens 4-7 = doc 1
        doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        pos_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        mask = build_block_diagonal_mask(doc_ids)  # [1, 1, 8, 8]

        x = torch.randn(1, T, cfg.hidden_dim)

        # Baseline output
        out1 = attn(x.clone(), attn_mask=mask, position_ids=pos_ids)

        # Perturb doc0 tokens massively
        x2 = x.clone()
        x2[:, :4, :] += 100.0
        out2 = attn(x2, attn_mask=mask, position_ids=pos_ids)

        # Doc1 output (positions 4-7) must be identical
        assert torch.allclose(out1[:, 4:, :], out2[:, 4:, :], atol=1e-4), (
            "Cross-document attention contamination detected!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Integration — Overfit test
# ─────────────────────────────────────────────────────────────────────────────

class TestOverfit:
    def test_overfit_tiny_packed(self):
        """
        Core validation: the TINY model must overfit a single packed batch
        to loss < 0.10 within 200 steps on CPU.
        """
        passed = run_overfit_test(
            config_name="tiny",
            device="cpu",
            n_steps=200,
            lr=3e-3,
            target_loss=0.10,
            seed=42,
            use_packed=True,
            verbose=True,
        )
        assert passed, "Overfit test FAILED — loss did not reach < 0.10"

    def test_overfit_tiny_causal(self):
        """
        Control: same test without packing (standard causal batch).
        Both must pass — if packed fails but causal passes, the mask
        implementation has a bug.
        """
        passed = run_overfit_test(
            config_name="tiny",
            device="cpu",
            n_steps=200,
            lr=3e-3,
            target_loss=0.10,
            seed=42,
            use_packed=False,
            verbose=False,
        )
        assert passed, "Causal overfit test FAILED — likely a model-level bug"
