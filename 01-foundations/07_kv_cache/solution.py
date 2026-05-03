"""KV Cache Solution — Reference implementation for efficient autoregressive generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KVCache:
    """Pre-allocated KV cache for efficient generation."""

    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, d_k: int):
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
        self.max_seq_len = max_seq_len

    def update(self, batch_size: int, start_pos: int, k: torch.Tensor, v: torch.Tensor):
        """Update cache at the given position."""
        seq_len = k.shape[2]
        self.k_cache[:batch_size, :, start_pos:start_pos + seq_len] = k
        self.v_cache[:batch_size, :, start_pos:start_pos + seq_len] = v

    def get(self, batch_size: int, end_pos: int):
        """Get cached K, V up to end_pos."""
        return (
            self.k_cache[:batch_size, :, :end_pos],
            self.v_cache[:batch_size, :, :end_pos],
        )


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

        # Compute Q, K, V projections and reshape for multi-head attention
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Update cache and retrieve full K, V if cache is provided
        if kv_cache is not None:
            kv_cache.update(B, start_pos, k, v)
            k, v = kv_cache.get(B, start_pos + T)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask: only attend to positions <= current position
        if T > 1:
            causal_mask = torch.triu(
                torch.ones(T, k.shape[2], device=x.device),
                diagonal=k.shape[2] - T + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(attn_output)
