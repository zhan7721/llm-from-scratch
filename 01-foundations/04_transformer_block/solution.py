"""Reference solution for the Transformer Block exercise."""

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
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS along the last dimension
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and apply learnable scale
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Formula: SwiGLU(x) = W2(SiLU(W_gate(x)) * W1(x))
    """

    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            # Default: 2/3 * 4 * d_model, rounded to nearest 256
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = ((d_ff + 255) // 256) * 256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path: SiLU activation on the gate projection
        gate = F.silu(self.w_gate(x))
        # Value path: linear projection
        value = self.w1(x)
        # Gated element-wise product, then project back
        return self.w2(gate * value)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, causal: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm attention with residual connection
        x = x + self.attn(self.attn_norm(x))
        # Pre-Norm FFN with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x
