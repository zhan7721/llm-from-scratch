"""Exercise: Implement the Transformer Block.

Complete the TODOs below to build RMSNorm, SwiGLU, and TransformerBlock.
Run `python tests.py` to verify your implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../03_attention"))
from attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    TODO: Implement the forward method.
      1. Compute the RMS: sqrt(mean of x^2 along the last dimension, plus eps)
      2. Normalize x by dividing by the RMS
      3. Multiply by the learnable weight

    Args:
        d_model: Dimensionality of the input.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Compute RMS (root mean square) along the last dimension
        # Hint: use x.pow(2).mean(-1, keepdim=True)
        rms = None  # YOUR CODE HERE

        # TODO: Normalize and scale
        return None  # YOUR CODE HERE


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Formula: SwiGLU(x) = W2(SiLU(W_gate(x)) * W1(x))

    TODO: Implement the forward method.
      1. Compute the gate: SiLU applied to w_gate(x)
      2. Compute the value: w1(x)
      3. Element-wise multiply gate * value
      4. Project back to d_model with w2

    Args:
        d_model: Input/output dimensionality.
        d_ff: Hidden dimension. If None, defaults to ~(2/3 * 4 * d_model).
    """

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = ((d_ff + 255) // 256) * 256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement SwiGLU
        # Hint: F.silu() is the SiLU/Swish activation
        return None  # YOUR CODE HERE


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    TODO: Implement the forward method.
      1. Apply RMSNorm, then attention, then add residual connection
      2. Apply RMSNorm, then FFN, then add residual connection

    Args:
        d_model: Dimensionality of the model.
        n_heads: Number of attention heads.
        d_ff: Hidden dimension for the FFN.
        causal: If True, apply causal masking in attention.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, causal: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pre-Norm attention with residual
        # x = x + self.attn(self.attn_norm(x))

        # TODO: Pre-Norm FFN with residual
        # x = x + self.ffn(self.ffn_norm(x))

        return x
