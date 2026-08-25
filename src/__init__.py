"""
src/__init__.py — Package init for the transformer source.
"""
from .config import ModelConfig, DEFAULT_300M, FLAGSHIP_700M, FLAGSHIP_1B, FLAGSHIP_3B, MAC_NANO, TINY_TEST
from .model import Transformer, RMSNorm, TransformerBlock
from .linear import GatedDeltaNet
from .moe import MoELayer
from .data import (
    SequencePacker,
    PackedDataset,
    build_block_diagonal_mask,
    packed_collate_fn,
    make_dataloader,
)

__all__ = [
    "ModelConfig",
    "DEFAULT_300M",
    "FLAGSHIP_700M",
    "FLAGSHIP_1B",
    "FLAGSHIP_3B",
    "MAC_NANO",
    "TINY_TEST",
    "Transformer",
    "RMSNorm",
    "TransformerBlock",
    "GatedDeltaNet",
    "MoELayer",
    "SequencePacker",
    "PackedDataset",
    "build_block_diagonal_mask",
    "packed_collate_fn",
    "make_dataloader",
]
