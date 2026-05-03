"""Token Embedding and Rotary Positional Embedding (RoPE) implementations."""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Maps token IDs to dense vectors.

    This is a simple lookup table that converts integer token IDs into
    continuous vectors of dimension d_model. The output is scaled by
    sqrt(d_model) to keep the magnitude of the embeddings consistent
    regardless of the embedding dimension.

    Args:
        vocab_size: Number of unique tokens in the vocabulary.
        d_model: Dimensionality of the embedding vectors.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            x: Tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            Embedding tensor with shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) as described in Su et al. 2021.

    RoPE encodes positional information by rotating query and key vectors
    in attention layers. Unlike absolute positional encodings, RoPE naturally
    captures relative positions through the rotation angle difference between
    two positions.

    The key idea: for a vector at position m, apply a rotation of angle
    m * theta_i to each pair of dimensions (2i, 2i+1), where theta_i is
    a frequency that depends on the dimension index.

    Args:
        d_model: Dimensionality of the vectors to rotate.
        max_seq_len: Maximum sequence length for precomputation.
        base: Base for the frequency computation (default 10000.0).
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        # Compute inverse frequencies: 1 / (base^(2i/d_model)) for i in [0, d_model/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _build_cache(self, seq_len: int, device: torch.device):
        """Precompute cosine and sine values for all positions and frequencies.

        Args:
            seq_len: Sequence length to compute for.
            device: Device to create tensors on.

        Returns:
            Tuple of (cos, sin) tensors, each with shape (seq_len, d_model).
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate frequencies to match d_model dimensions
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions by 90 degrees.

        For input [..., x1, x2], produces [..., -x2, x1].

        This is equivalent to multiplying each 2D vector by the
        90-degree rotation matrix [[0, -1], [1, 0]].

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Rotated tensor of same shape.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor.

        The rotation is applied as:
            x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)

        This is equivalent to applying a rotation matrix to each pair
        of dimensions, where the angle depends on the position and
        the dimension pair.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Rotated tensor of same shape.
        """
        seq_len = x.shape[1]
        cos, sin = self._build_cache(seq_len, x.device)
        cos = cos[:seq_len].unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin
