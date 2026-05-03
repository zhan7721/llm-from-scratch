"""Multi-Head Attention and Grouped Query Attention — Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionSolution(nn.Module):
    """Standard Multi-Head Attention (MHA) — solution version.

    Args:
        d_model: Total dimensionality of the model.
        n_heads: Number of attention heads. Must divide d_model evenly.
        causal: If True, apply a causal mask.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Run multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional mask, broadcastable to
                (batch_size, n_heads, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape

        # 1. Project to Q, K, V and reshape to (B, n_heads, T, d_k)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Apply causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # 4. Apply optional mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # 6. Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attn_output)


class GroupedQueryAttentionSolution(nn.Module):
    """Grouped Query Attention (GQA) — solution version.

    Args:
        d_model: Total dimensionality of the model.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        causal: If True, apply a causal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.causal = causal

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads.

        Args:
            x: Tensor of shape (B, n_kv_heads, T, d_k).

        Returns:
            Tensor of shape (B, n_heads, T, d_k).
        """
        B, n_kv, T, d_k = x.shape
        if self.n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]
            .expand(B, n_kv, self.n_rep, T, d_k)
            .reshape(B, self.n_heads, T, d_k)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Run grouped query attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional mask, same semantics as MHA.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape

        # 1. Project and reshape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 2. Expand KV heads
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # 3. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 4. Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # 5. Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attn_output)
