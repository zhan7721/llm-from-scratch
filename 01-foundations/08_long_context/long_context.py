"""Long context techniques: RoPE scaling and YaRN.

This module implements methods for extending the context window of transformer
models that use Rotary Position Embeddings (RoPE). When a model is trained with
a fixed context length (e.g., 4096 tokens), these techniques allow it to handle
longer sequences at inference time, often with minimal quality degradation.

Techniques covered:
- ScaledRoPE: Linear scaling of position indices (used in LLaMA 2 Long)
- YaRNRope: NTK-aware scaling with attention temperature (Peng et al. 2023)
"""

import torch
import torch.nn as nn
import math


class ScaledRoPE(nn.Module):
    """RoPE with linear scaling for extended context length.

    Scales the position indices by a factor to extend the effective context
    window. For example, with scale_factor=2.0, a model trained on 4096 tokens
    can handle 8192 tokens.

    This approach was used in LLaMA 2 Long. The idea is simple: divide all
    position indices by the scale factor, effectively "squashing" the position
    encodings so that more positions fit within the original frequency range.

    Args:
        d_model: Dimensionality of the model (must be even).
        max_seq_len: Maximum sequence length the model was trained on.
        base: Base frequency for computing inverse frequencies (theta in RoPE).
        scale_factor: Factor to scale positions by. A value of 2.0 means the
            effective context is doubled (positions are halved).
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0, scale_factor: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor

        # Compute inverse frequencies: theta_i = 1 / (base^(2i/d_model))
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the input by swapping and negating halves.

        For a vector [x1, x2], returns [-x2, x1]. This is the core operation
        of RoPE that creates the rotation matrix effect.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled RoPE to the input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Rotated tensor of the same shape.
        """
        seq_len = x.shape[1]
        # Scale positions: divide by scale_factor to compress position range
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) / self.scale_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


class YaRNRope(nn.Module):
    """YaRN (Yet another RoPE extensioN) for context extension.

    Combines NTK-aware scaling with attention temperature scaling to extend
    the context window while preserving the model's understanding of both
    local and global position relationships.

    Key innovations over simple linear scaling:
    1. NTK-aware scaling: Adjusts the base frequency instead of position
       indices, preserving high-frequency components that encode local context.
    2. Attention temperature: Scales down attention logits to compensate for
       the changed frequency distribution.

    Reference: Peng et al. 2023, "YaRN: Efficient Context Window Extension
    of Large Language Models"

    Args:
        d_model: Dimensionality of the model (must be even).
        max_seq_len: Maximum sequence length the model was trained on.
        base: Base frequency for computing inverse frequencies.
        scale_factor: Factor to extend context by (e.g., 4.0 for 4x extension).
        beta_fast: Tuning parameter for NTK-aware scaling (higher = more
            preservation of high frequencies). Default: 32.0.
        beta_slow: Tuning parameter for NTK-aware scaling (lower = more
            aggressive low-frequency modification). Default: 1.0.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0,
                 scale_factor: float = 4.0, beta_fast: float = 32.0, beta_slow: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor

        # NTK-aware scaling: adjust base frequency instead of position indices.
        # This preserves high-frequency components (local context) while
        # extending low-frequency components (global context).
        base_scaled = base * (scale_factor ** (d_model / (d_model - 2)))

        inv_freq = 1.0 / (base_scaled ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Attention temperature: scale down logits to compensate for the
        # changed frequency distribution after NTK-aware scaling.
        self.attn_factor = 1.0 / math.sqrt(scale_factor)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the input by swapping and negating halves."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply YaRN-scaled RoPE to the input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Rotated and temperature-scaled tensor of the same shape.
        """
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        return (x * cos + self._rotate_half(x) * sin) * self.attn_factor
