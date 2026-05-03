"""KV Cache Exercise — Implement efficient autoregressive generation.

Your task: Complete the KVCache and CachedAttention classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KVCache:
    """Pre-allocated KV cache for efficient generation."""

    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, d_k: int):
        # TODO: Initialize k_cache and v_cache as zero tensors
        # Shape: (max_batch_size, n_heads, max_seq_len, d_k)
        # Also store max_seq_len as an attribute
        pass

    def update(self, batch_size: int, start_pos: int, k: torch.Tensor, v: torch.Tensor):
        """Update cache at the given position.

        Args:
            batch_size: Number of sequences in the batch.
            start_pos: Position to start writing in the cache.
            k: Key tensor of shape (batch, n_heads, seq_len, d_k).
            v: Value tensor of shape (batch, n_heads, seq_len, d_k).
        """
        # TODO: Store k and v into the cache at position start_pos:start_pos+seq_len
        # Only update the first batch_size entries
        pass

    def get(self, batch_size: int, end_pos: int):
        """Get cached K, V up to end_pos.

        Args:
            batch_size: Number of sequences in the batch.
            end_pos: Position to read up to (exclusive).

        Returns:
            Tuple of (k_cache, v_cache) tensors.
        """
        # TODO: Return cached k and v for the first batch_size entries up to end_pos
        pass


class CachedAttention(nn.Module):
    """Multi-Head Attention with KV Cache support."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: KVCache = None, start_pos: int = 0) -> torch.Tensor:
        """Forward pass with optional KV cache.

        Args:
            x: Input (batch, seq_len, d_model). During generation, seq_len=1.
            kv_cache: KV cache object for incremental decoding.
            start_pos: Current position in the sequence.
        """
        B, T, _ = x.shape

        # TODO: Compute Q, K, V projections
        # Reshape to (B, n_heads, T, d_k) for multi-head attention
        q = None  # TODO
        k = None  # TODO
        v = None  # TODO

        # TODO: If kv_cache is provided, update it with new k, v
        # Then retrieve full k, v from cache

        # TODO: Compute scaled dot-product attention
        # scores = q @ k^T / sqrt(d_k)
        scores = None  # TODO

        # TODO: Apply causal mask when T > 1
        # Use torch.triu to create upper triangular mask

        # TODO: Apply softmax and compute output
        attn_output = None  # TODO

        # TODO: Reshape and apply output projection
        return None  # TODO
