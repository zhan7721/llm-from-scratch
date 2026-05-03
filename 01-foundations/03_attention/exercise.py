"""Multi-Head Attention and Grouped Query Attention — Exercise.

Fill in the TODO methods to implement working attention layers.
Start with MultiHeadAttention, then move to GroupedQueryAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionExercise(nn.Module):
    """Standard Multi-Head Attention (MHA).

    Splits Q, K, V into multiple heads, computes attention independently
    in each head, then concatenates and projects.

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

        # TODO: Create four linear projections for Q, K, V, and output
        # HINT: Each should map d_model -> d_model, with no bias.
        #       self.W_q = nn.Linear(d_model, d_model, bias=False)
        #       ... same for W_k, W_v, W_o
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Run multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional mask of shape broadcastable to
                (batch_size, n_heads, seq_len, seq_len).
                Positions with value 0 are masked out.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        Steps:
            1. Project x to Q, K, V using W_q, W_k, W_v
            2. Reshape each from (B, T, d_model) to (B, n_heads, T, d_k)
               - Use .view(B, T, self.n_heads, self.d_k).transpose(1, 2)
            3. Compute attention scores: Q @ K^T / sqrt(d_k)
            4. If causal, create and apply causal mask (upper triangular)
            5. If mask provided, apply it
            6. Apply softmax to get attention weights
            7. Multiply by V to get context
            8. Reshape back to (B, T, d_model) and project with W_o
        """
        raise NotImplementedError("TODO: implement forward")


class GroupedQueryAttentionExercise(nn.Module):
    """Grouped Query Attention (GQA).

    Like MHA, but K and V have fewer heads than Q. Each KV head is
    shared among a group of query heads.

    Args:
        d_model: Total dimensionality of the model.
        n_heads: Number of query heads. Must divide d_model evenly.
        n_kv_heads: Number of KV heads. Must divide n_heads evenly.
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

        # TODO: Create linear projections
        # HINT: Q projects to n_heads * d_k
        #       K and V project to n_kv_heads * d_k (fewer heads!)
        #       W_o projects back to d_model
        pass

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads.

        Args:
            x: Tensor of shape (B, n_kv_heads, T, d_k).

        Returns:
            Tensor of shape (B, n_heads, T, d_k).

        HINT:
            1. If n_rep == 1, return x unchanged
            2. Otherwise, insert a new dimension and expand:
               x[:, :, None, :, :]  -> (B, n_kv, 1, T, d_k)
               .expand(B, n_kv, n_rep, T, d_k)
               .reshape(B, n_heads, T, d_k)
        """
        raise NotImplementedError("TODO: implement _repeat_kv")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Run grouped query attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional mask, same semantics as MHA.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).

        Steps:
            1. Project x to Q (n_heads), K (n_kv_heads), V (n_kv_heads)
            2. Reshape: Q -> (B, n_heads, T, d_k),
                        K -> (B, n_kv_heads, T, d_k),
                        V -> (B, n_kv_heads, T, d_k)
            3. Expand K and V using _repeat_kv
            4. Compute attention scores: Q @ K^T / sqrt(d_k)
            5. Apply causal mask and/or provided mask
            6. Softmax and multiply by V
            7. Reshape and project with W_o
        """
        raise NotImplementedError("TODO: implement forward")
