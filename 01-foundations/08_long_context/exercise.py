"""Long Context Exercise — Implement RoPE scaling and YaRN.

Your task: Complete the ScaledRoPE and YaRNRope classes to extend
the context window of a transformer model.
"""

import torch
import torch.nn as nn
import math


class ScaledRoPE(nn.Module):
    """RoPE with linear scaling for extended context length.

    Scales the position indices by a factor to extend the effective context window.
    Used in LLaMA 2 Long.

    Args:
        d_model: Dimensionality of the model (must be even).
        max_seq_len: Maximum sequence length the model was trained on.
        base: Base frequency for computing inverse frequencies.
        scale_factor: Factor to scale positions by (e.g., 2.0 doubles context).
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
        """Rotate the input by swapping and negating halves."""
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

        # TODO: Create position indices scaled by scale_factor
        # Hint: divide positions by self.scale_factor
        t = None  # TODO

        # TODO: Compute frequency embeddings
        # Hint: outer product of t and inv_freq, then cat with itself
        freqs = None  # TODO
        emb = None  # TODO

        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)

        # TODO: Apply RoPE rotation
        # Hint: x * cos + rotate_half(x) * sin
        return None  # TODO


class YaRNRope(nn.Module):
    """YaRN (Yet another RoPE extensioN) for context extension.

    Combines NTK-aware scaling with attention temperature scaling.
    Reference: Peng et al. 2023

    Args:
        d_model: Dimensionality of the model (must be even).
        max_seq_len: Maximum sequence length the model was trained on.
        base: Base frequency for computing inverse frequencies.
        scale_factor: Factor to extend context by (e.g., 4.0 for 4x extension).
        beta_fast: Tuning parameter for NTK-aware scaling.
        beta_slow: Tuning parameter for NTK-aware scaling.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0,
                 scale_factor: float = 4.0, beta_fast: float = 32.0, beta_slow: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor

        # TODO: Implement NTK-aware scaling
        # Instead of scaling positions, we scale the base frequency.
        # Formula: base_scaled = base * (scale_factor ^ (d_model / (d_model - 2)))
        # This preserves high-frequency components (local context) while
        # extending low-frequency components (global context).
        base_scaled = None  # TODO

        inv_freq = 1.0 / (base_scaled ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # TODO: Compute attention temperature factor
        # Hint: 1 / sqrt(scale_factor)
        self.attn_factor = None  # TODO

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

        # TODO: Apply RoPE rotation with attention temperature scaling
        # Hint: (x * cos + rotate_half(x) * sin) * attn_factor
        return None  # TODO
