"""
data.py — Sequence packing data pipeline with block-diagonal attention masking.

Problem we solve:
    Naive padding wastes compute: a 2048-token context filled with a
    64-token doc + 1984 padding tokens still runs the full attention.
    Sequence packing concatenates multiple documents into a single
    context window, then uses a block-diagonal attention mask to ensure
    tokens from document A never attend to tokens from document B.

Implementation:
    SequencePacker:
        - Accepts a list of token-ID lists (documents).
        - Greedily packs them into fixed-length blocks of `block_size`.
        - Returns packed input_ids, targets, position_ids, and doc_ids.

    build_block_diagonal_mask(doc_ids):
        - Given a [batch, seq_len] tensor of document IDs per position,
          constructs an additive attention mask [batch, 1, seq_len, seq_len]
          where positions from different documents get -inf (masked out).
        - Causal masking is baked in simultaneously.

    PackedDataset:
        - A torch.utils.data.Dataset wrapping SequencePacker.
        - Returns (input_ids, targets, position_ids, attn_mask) tuples
          ready to pass directly to Transformer.forward().

Example:
    doc1 = [1, 2, 3, 4]      # 4 tokens
    doc2 = [5, 6, 7]          # 3 tokens
    packed = [1, 2, 3, 4, 5, 6, 7, <pad>]   (block_size=8)
    doc_ids= [0, 0, 0, 0, 1, 1, 1, -1]

    attention mask ensures token at position 4 (first token of doc2)
    does NOT attend to any position 0-3 (doc1 tokens).
"""

from __future__ import annotations
import random
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ── Mask construction ─────────────────────────────────────────────────────────

def build_block_diagonal_mask(
    doc_ids: Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Build a causal block-diagonal additive attention mask.

    Args:
        doc_ids : [batch, seq_len] — integer doc ID per position.
                  Padding positions should have doc_id = -1.
        dtype   : Output dtype (float32 or bfloat16). The mask contains
                  0.0 (attend) or -inf (do not attend).

    Returns:
        mask : [batch, 1, seq_len, seq_len]
               mask[b, 0, i, j] = 0.0   if j <= i AND doc_ids[b,i] == doc_ids[b,j]
                                          AND doc_ids[b,j] != -1
               mask[b, 0, i, j] = -inf  otherwise

    This mask is passed directly to F.scaled_dot_product_attention as
    attn_mask (additive), replacing the is_causal shortcut.
    """
    B, T = doc_ids.shape
    device = doc_ids.device

    # ── Same-document mask: [B, T, T]  (True where i and j share a doc) ──
    # Broadcast doc_ids[:, :, None] vs doc_ids[:, None, :]
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # [B, T, T]

    # ── Padding mask: exclude positions with doc_id == -1 ────────────────
    valid_j = (doc_ids != -1).unsqueeze(1)  # [B, 1, T]
    same_doc = same_doc & valid_j            # [B, T, T]

    # ── Causal mask: only attend to j <= i ───────────────────────────────
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))  # [T, T]
    combined = same_doc & causal.unsqueeze(0)                                 # [B, T, T]

    # ── Convert to additive float mask ───────────────────────────────────
    # 0.0 where allowed, -inf where blocked
    mask = torch.zeros(B, T, T, device=device, dtype=dtype)
    mask = mask.masked_fill(~combined, float("-inf"))

    return mask.unsqueeze(1)  # [B, 1, T, T]


# ── Sequence Packer ───────────────────────────────────────────────────────────

class SequencePacker:
    """
    Greedy document packer.

    Packs variable-length token sequences into fixed-size blocks.
    Documents are NOT split across block boundaries (whole docs only).
    A document longer than block_size is truncated with a warning.

    Attributes:
        block_size : int
        pad_id     : int  — token ID used for padding (default 0)
    """

    def __init__(self, block_size: int, pad_id: int = 0) -> None:
        self.block_size = block_size
        self.pad_id = pad_id

    def pack(
        self,
        documents: list[list[int]],
        shuffle: bool = True,
        seed: int = 42,
    ) -> list[dict]:
        """
        Pack documents into blocks.

        Args:
            documents : list of token-id lists
            shuffle   : shuffle document order before packing
            seed      : RNG seed for reproducibility

        Returns:
            List of dicts, each containing:
                input_ids   : list[int], length == block_size
                doc_ids     : list[int], doc index per position (-1 = pad)
                position_ids: list[int], position within doc per token
        """
        rng = random.Random(seed)
        docs = [list(d) for d in documents]  # copy

        # Truncate oversized documents
        truncated = 0
        for i, doc in enumerate(docs):
            if len(doc) > self.block_size:
                docs[i] = doc[:self.block_size]
                truncated += 1
        if truncated:
            print(f"[SequencePacker] WARNING: truncated {truncated} document(s) > block_size")

        # Filter empty
        docs = [d for d in docs if len(d) > 0]

        if shuffle:
            rng.shuffle(docs)

        blocks: list[dict] = []
        current_ids: list[int] = []
        current_doc_ids: list[int] = []
        current_pos_ids: list[int] = []
        doc_counter = 0

        for doc in docs:
            doc_len = len(doc)
            # If this document doesn't fit, flush current block
            if len(current_ids) + doc_len > self.block_size:
                if current_ids:
                    blocks.append(self._pad_block(
                        current_ids, current_doc_ids, current_pos_ids
                    ))
                current_ids = []
                current_doc_ids = []
                current_pos_ids = []
                doc_counter = 0

            current_ids.extend(doc)
            current_doc_ids.extend([doc_counter] * doc_len)
            current_pos_ids.extend(list(range(doc_len)))
            doc_counter += 1

        # Final partial block
        if current_ids:
            blocks.append(self._pad_block(
                current_ids, current_doc_ids, current_pos_ids
            ))

        return blocks

    def _pad_block(
        self,
        ids: list[int],
        doc_ids: list[int],
        pos_ids: list[int],
    ) -> dict:
        """Pad a partial block to block_size."""
        pad_len = self.block_size - len(ids)
        return {
            "input_ids":    ids + [self.pad_id] * pad_len,
            "doc_ids":      doc_ids + [-1] * pad_len,
            "position_ids": pos_ids + [0] * pad_len,
        }


# ── Dataset ───────────────────────────────────────────────────────────────────

class PackedDataset(Dataset):
    """
    PyTorch Dataset wrapping the SequencePacker.

    Each item is a dict with:
        input_ids   : [block_size]  (LongTensor)
        targets     : [block_size, K] (LongTensor) — inputs shifted by k+1,
                                                     padded/cross-doc = -100
        position_ids: [block_size]  (LongTensor)
        doc_ids     : [block_size]  (LongTensor)

    The attention mask is NOT stored per-sample; it's constructed in the
    collate function so it benefits from batching (can be batched together).
    """

    def __init__(
        self,
        documents: list[list[int]],
        block_size: int,
        pad_id: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        num_mtp_heads: int = 0,
    ) -> None:
        self.block_size = block_size
        self.pad_id = pad_id
        self.num_mtp_heads = num_mtp_heads
        packer = SequencePacker(block_size=block_size, pad_id=pad_id)
        self.blocks = packer.pack(documents, shuffle=shuffle, seed=seed)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        block = self.blocks[idx]
        input_ids     = torch.tensor(block["input_ids"],    dtype=torch.long)
        doc_ids       = torch.tensor(block["doc_ids"],      dtype=torch.long)
        # within-doc positions (0-indexed per document): stored for reference / future use
        doc_positions = torch.tensor(block["position_ids"], dtype=torch.long)

        # Absolute positions for RoPE: simply 0, 1, 2, ..., block_size - 1.
        # The cross-document boundary is handled entirely by the block-diagonal
        # attention mask (derived from doc_ids), NOT by resetting RoPE positions.
        # Using absolute positions keeps the RoPE cache indexing simple and valid.
        position_ids = torch.arange(self.block_size, dtype=torch.long)

        # Targets = input shifted left by k; padded and cross-doc masked = -100
        K = 1 + self.num_mtp_heads
        targets = torch.full((self.block_size, K), -100, dtype=torch.long)
        
        for k in range(K):
            shift = k + 1
            if shift < self.block_size:
                # Valid mask: within the same document AND not padding
                valid_mask = (doc_ids[:-shift] == doc_ids[shift:]) & (doc_ids[:-shift] != -1)
                targets[:-shift, k] = torch.where(
                    valid_mask,
                    input_ids[shift:],
                    torch.tensor(-100, dtype=torch.long)
                )

        if K == 1:
            targets = targets.squeeze(-1)

        return {
            "input_ids":    input_ids,
            "targets":      targets,
            "position_ids": position_ids,      # absolute, for RoPE
            "doc_ids":      doc_ids,
            "doc_positions": doc_positions,    # within-doc, for reference
        }


# ── Collate function ──────────────────────────────────────────────────────────

def packed_collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """
    Custom collate function for PackedDataset.

    Stacks tensors and constructs the block-diagonal attention mask
    for the entire batch in one vectorized call.
    """
    input_ids     = torch.stack([b["input_ids"]     for b in batch])
    targets       = torch.stack([b["targets"]       for b in batch])
    position_ids  = torch.stack([b["position_ids"]  for b in batch])
    doc_ids       = torch.stack([b["doc_ids"]       for b in batch])
    doc_positions = torch.stack([b["doc_positions"] for b in batch])

    # Build block-diagonal causal mask
    attn_mask = build_block_diagonal_mask(doc_ids, dtype=torch.float32)

    return {
        "input_ids":    input_ids,
        "targets":      targets,
        "position_ids": position_ids,
        "doc_ids":      doc_ids,
        "doc_positions": doc_positions,
        "attn_mask":    attn_mask,
    }


def make_dataloader(
    documents: list[list[int]],
    block_size: int,
    batch_size: int,
    pad_id: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    num_mtp_heads: int = 0,
) -> DataLoader:
    """
    Convenience function: build a DataLoader with packed sequences.

    Args:
        documents  : list of token-id lists
        block_size : tokens per context window
        batch_size : samples per batch
        pad_id     : token ID for padding (should not appear in vocab)
        shuffle    : shuffle document order
        seed       : reproducibility seed
        num_workers: DataLoader worker processes
        num_mtp_heads: Number of extra multi-token prediction heads

    Returns:
        DataLoader yielding batches of:
            input_ids    [B, T]
            targets      [B, T, K] or [B, T] if K=1
            position_ids [B, T]
            doc_ids      [B, T]
            attn_mask    [B, 1, T, T]
    """
    dataset = PackedDataset(
        documents=documents,
        block_size=block_size,
        pad_id=pad_id,
        shuffle=shuffle,
        seed=seed,
        num_mtp_heads=num_mtp_heads,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,   # Dataset-level shuffle handles order
        num_workers=num_workers,
        collate_fn=packed_collate_fn,
        # pin_memory only works on CUDA; MPS and CPU don't support it
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
