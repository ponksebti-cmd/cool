"""
tests/test_stage3.py — Unit tests for Stage 3 (Sparse MoE).

Run with: pytest tests/test_stage3.py -v
"""

from __future__ import annotations
import sys
import torch
import torch.nn as nn
import pytest
import math

sys.path.insert(0, ".")
from src.config import ModelConfig, TINY_TEST
from src.moe import MoELayer
from src.model import Transformer

@pytest.fixture
def cfg() -> ModelConfig:
    return TINY_TEST


class TestMoELayer:
    def test_output_shape(self, cfg):
        moe = MoELayer(cfg, layer_idx=0)
        x = torch.randn(2, 16, cfg.hidden_dim)
        out = moe(x)
        assert out.shape == (2, 16, cfg.hidden_dim)

    def test_aux_loss_uniform(self, cfg):
        """If routing is perfectly uniform, L_aux should be exactly 1.0."""
        moe = MoELayer(cfg, layer_idx=0)
        
        # Override router weights to 0 so all logits are 0 -> probs are uniform 1/E
        nn.init.zeros_(moe.router.weight)
        
        x = torch.randn(2, 16, cfg.hidden_dim)
        out = moe(x)
        
        # P_i = 1/E. f_i = 1/E. E * sum(1/E * 1/E) = E * E * (1/E^2) = 1.0
        assert math.isclose(moe.l_aux.item(), 1.0, rel_tol=1e-4)

    def test_aux_loss_skewed(self, cfg):
        """If routing is completely skewed, L_aux should equal E / K."""
        moe = MoELayer(cfg, layer_idx=0)
        
        # Force huge logits for expert 0, negative for others
        with torch.no_grad():
            moe.router.weight.fill_(-100.0)
            moe.router.weight[0].fill_(100.0)
            
        x = torch.ones(2, 16, cfg.hidden_dim) # positive input
        out = moe(x)
        
        # Expert 0 gets all 1st choices, expert 1 gets 2nd choices.
        # f_0 = 1/K, P_0 = 1.0
        # L_aux = E * (1/K * 1.0) = E / K
        expected = float(cfg.n_routed_experts) / cfg.n_active_experts
        assert math.isclose(moe.l_aux.item(), expected, rel_tol=1e-4)

    def test_capacity_dropping(self, cfg):
        """Test that tokens exceeding capacity are dropped."""
        # Use a strict capacity factor
        cfg.moe_capacity_factor = 1.0
        cfg.n_active_experts = 2
        moe = MoELayer(cfg, layer_idx=0)
        
        # N = 32 tokens
        B, T = 2, 16
        N = B * T
        
        # Force all tokens to prefer expert 0 and 1
        with torch.no_grad():
            moe.router.weight.fill_(-100.0)
            moe.router.weight[0].fill_(100.0)
            moe.router.weight[1].fill_(50.0)
            
        x = torch.ones(B, T, cfg.hidden_dim)
        out = moe(x)
        
        # Capacity formula: int((N * top_k / E) * factor)
        # N=32, top_k=2, E=4 (TINY_TEST usually doesn't define E, let's check defaults)
        # TINY default n_routed_experts is 8
        capacity = int((N * cfg.n_active_experts / cfg.n_routed_experts) * cfg.moe_capacity_factor)
        
        # Expect expert 0 and 1 to hit the capacity limit exactly
        counts = moe.expert_counts
        assert counts[0].item() == capacity
        assert counts[1].item() == capacity
        # Others should be 0
        assert counts[2:].sum().item() == 0

class TestMoETransformer:
    def test_forward_with_moe(self, cfg):
        """Forward pass should succeed and accumulate aux loss."""
        model = Transformer(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)
        
        # Check aux loss exists
        aux_loss = 0.0
        for layer in model.layers:
            if hasattr(layer.mlp, "l_aux"):
                aux_loss += layer.mlp.l_aux
                
        assert aux_loss > 0.0
