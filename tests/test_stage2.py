"""
tests/test_stage2.py — Unit tests for Stage 2 (Hybrid Attention).

Run with: pytest tests/test_stage2.py -v
"""

from __future__ import annotations
import sys
import torch
import pytest

sys.path.insert(0, ".")
from src.config import ModelConfig, TINY_TEST
from src.linear import GatedDeltaNet
from src.model import Transformer
from src.attention import GroupedQueryAttention
from src.data import build_block_diagonal_mask


@pytest.fixture
def cfg() -> ModelConfig:
    return TINY_TEST


class TestGatedDeltaNet:
    def test_output_shape(self, cfg):
        delta = GatedDeltaNet(cfg, layer_idx=0)
        x = torch.randn(2, 16, cfg.hidden_dim)
        out = delta(x)
        assert out.shape == (2, 16, cfg.hidden_dim)

    def test_causal_property(self, cfg):
        """Token at position i should not depend on position j > i."""
        delta = GatedDeltaNet(cfg, layer_idx=0)
        delta.eval()
        T = 8
        x = torch.randn(1, T, cfg.hidden_dim)

        out1 = delta(x.clone(), is_causal=True)[:, 3, :]

        # Perturb future position
        x2 = x.clone()
        x2[:, 5, :] += 10.0
        out2 = delta(x2, is_causal=True)[:, 3, :]

        assert torch.allclose(out1, out2, atol=1e-5), (
            "GatedDeltaNet causal violation"
        )

    def test_explicit_mask(self, cfg):
        """Cross-document masking should block flow."""
        delta = GatedDeltaNet(cfg, layer_idx=0)
        delta.eval()
        T = 8
        # doc0 = pos 0..3, doc1 = pos 4..7
        doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        mask = build_block_diagonal_mask(doc_ids)

        x = torch.randn(1, T, cfg.hidden_dim)
        out1 = delta(x.clone(), attn_mask=mask)

        # Perturb doc0 massively
        x2 = x.clone()
        x2[:, :4, :] += 100.0
        out2 = delta(x2, attn_mask=mask)

        # doc1 outputs should remain identical
        assert torch.allclose(out1[:, 4:, :], out2[:, 4:, :], atol=1e-4)


class TestHybridModel:
    def test_layer_instantiation(self, cfg):
        """Ensure the 3:1 pattern (or whatever config sets) is respected."""
        # Force a specific pattern for testing
        cfg.attn_layer_period = 3
        cfg.n_layers = 6
        model = Transformer(cfg)
        
        # Period = 3, so layer_idx + 1 % 3 == 0 means Dense
        # Layers 0, 1 -> Linear
        # Layer 2 -> Dense
        # Layers 3, 4 -> Linear
        # Layer 5 -> Dense
        
        assert isinstance(model.layers[0].attn, GatedDeltaNet)
        assert isinstance(model.layers[1].attn, GatedDeltaNet)
        assert isinstance(model.layers[2].attn, GroupedQueryAttention)
        assert isinstance(model.layers[3].attn, GatedDeltaNet)
        assert isinstance(model.layers[4].attn, GatedDeltaNet)
        assert isinstance(model.layers[5].attn, GroupedQueryAttention)

    def test_forward_hybrid(self, cfg):
        """Forward pass should succeed without shape mismatches."""
        model = Transformer(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_residual_scaling_applies_to_deltanet(self, cfg):
        """Ensure DeltaNet's o_proj gets the DeepNet 1/sqrt(2L) scale."""
        import copy
        torch.manual_seed(1)
        cfg.scale_residual_init = True
        m_scaled = Transformer(cfg)
        
        torch.manual_seed(1)
        cfg2 = copy.deepcopy(cfg)
        cfg2.scale_residual_init = False
        m_unscaled = Transformer(cfg2)
        
        # Check a Linear layer
        assert isinstance(m_scaled.layers[0].attn, GatedDeltaNet)
        o_scaled = m_scaled.layers[0].attn.o_proj.weight.norm().item()
        o_unscaled = m_unscaled.layers[0].attn.o_proj.weight.norm().item()
        
        assert o_scaled < o_unscaled
