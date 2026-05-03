"""Transformer Block with Pre-Norm, SwiGLU FFN, and Residual connections.

This module implements the core building block of modern LLMs (LLaMA, PaLM, etc.):
- RMSNorm: Root Mean Square Layer Normalization (simpler than LayerNorm)
- SwiGLU: Gated feed-forward network (replaces ReLU FFN)
- TransformerBlock: Pre-Norm residual block combining attention + FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../03_attention"))
from attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style).

    Unlike LayerNorm, RMSNorm does not center the input (no mean subtraction).
    It only normalizes by the root mean square, then applies a learnable scale.
    This is cheaper to compute and has been shown to perform equally well.

    Formula:
        RMSNorm(x) = x / RMS(x) * weight
        where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        d_model: Dimensionality of the input.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of the same shape.
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network as in PaLM/LLaMA.

    Standard Transformer FFN uses ReLU activation:
        FFN(x) = W2 * ReLU(W1 * x)

    SwiGLU replaces this with a gated mechanism:
        SwiGLU(x) = W2 * (SiLU(W_gate * x) * W1 * x)

    The SiLU (Swish) activation acts as a gate on the linear projection,
    producing a more expressive nonlinearity. The hidden dimension is
    typically set to (2/3) * 4 * d_model rounded to a multiple of 256.

    Args:
        d_model: Input/output dimensionality.
        d_ff: Hidden dimension. If None, defaults to (2/3 * 4 * d_model)
              rounded up to the nearest multiple of 256.
    """

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            # Default: 2/3 * 4 * d_model, rounded to nearest multiple of 256
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = ((d_ff + 255) // 256) * 256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block (LLaMA-style).

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    This is a Pre-Norm design where normalization is applied before each
    sub-layer (attention and FFN), not after. Pre-Norm is preferred in
    modern LLMs because it:
    - Stabilizes training (gradients flow more smoothly)
    - Enables training of deeper models without warmup
    - Allows higher learning rates

    Args:
        d_model: Dimensionality of the model.
        n_heads: Number of attention heads.
        d_ff: Hidden dimension for the FFN. If None, uses the default
              SwiGLU formula.
        causal: If True, apply causal masking in attention.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, causal: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
