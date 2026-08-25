"""
tests/test_stage4.py — Unit tests for Stage 4 (Multi-Token Prediction).

Run with: pytest tests/test_stage4.py -v
"""

from __future__ import annotations
import sys
import torch
import pytest

sys.path.insert(0, ".")
from src.config import ModelConfig, TINY_TEST
from src.model import Transformer
from src.data import PackedDataset


import copy

@pytest.fixture
def cfg() -> ModelConfig:
    cfg_mtp = copy.deepcopy(TINY_TEST)
    cfg_mtp.num_mtp_heads = 2
    return cfg_mtp


class TestMTPDataset:
    def test_mtp_targets_shape(self):
        """Ensure targets tensor has shape [block_size, 1+num_mtp_heads]."""
        # Pass a raw document of length 10
        docs = [list(range(10))]
        dataset = PackedDataset(docs, block_size=10, num_mtp_heads=2, shuffle=False)
        item = dataset[0]
        
        targets = item["targets"]
        assert targets.shape == (10, 3)  # K = 1 + 2

    def test_mtp_cross_doc_masking(self):
        """Ensure targets that cross document boundaries are masked out (-100)."""
        # doc0: 0,1,2. doc1: 3,4,5. 
        docs = [[10, 11, 12], [13, 14, 15]]
        # Packing into a block_size of 6 will perfectly fit them both
        dataset = PackedDataset(docs, block_size=6, num_mtp_heads=2, shuffle=False)
        targets = dataset[0]["targets"]
        
        # Base targets (shift 1)
        # 10->11, 11->12, 12->-100(cross doc), 13->14, 14->15, 15->-100(end)
        assert targets[0, 0] == 11
        assert targets[2, 0] == -100
        assert targets[5, 0] == -100
        
        # MTP target 1 (shift 2)
        # 10->12, 11->-100(cross), 12->-100(cross), 13->15, 14->-100, 15->-100
        assert targets[0, 1] == 12
        assert targets[1, 1] == -100
        assert targets[2, 1] == -100
        assert targets[3, 1] == 15
        
        # MTP target 2 (shift 3)
        # doc len is 3, so a shift of 3 ALWAYS crosses a doc boundary or falls off
        assert torch.all(targets[:, 2] == -100)


class TestMTPTransformer:
    def test_forward_with_mtp_heads(self, cfg):
        """Forward pass should succeed and integrate MTP loss."""
        model = Transformer(cfg)
        
        # 2 heads + base lm_head -> 3 total loss terms
        assert model.num_mtp_heads == 2
        assert len(model.mtp_projections) == 2
        
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        
        # Create fake targets of shape [B, T, 3]
        targets = torch.randint(0, cfg.vocab_size, (2, 16, 3))
        
        logits, loss = model(x, targets=targets)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_legacy_2d_targets(self, cfg):
        """MTP model should still gracefully handle [B, T] targets without crashing."""
        model = Transformer(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        targets = torch.randint(0, cfg.vocab_size, (2, 16))
        
        logits, loss = model(x, targets=targets)
        # MTP logic should be skipped because targets only has 1 dim (base)
        assert loss is not None
