"""Token Embedding and Rotary Positional Embedding — Exercise.

Fill in the TODO methods to implement working embedding layers.
Start with TokenEmbedding, then move to RotaryPositionalEmbedding.
"""

import torch
import torch.nn as nn
import math


class TokenEmbeddingExercise(nn.Module):
    """Maps token IDs to dense vectors.

    This is a simple lookup table that converts integer token IDs into
    continuous vectors of dimension d_model.

    Args:
        vocab_size: Number of unique tokens in the vocabulary.
        d_model: Dimensionality of the embedding vectors.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # TODO: Create an nn.Embedding layer
        # HINT: nn.Embedding(vocab_size, d_model) creates a lookup table
        #       mapping each token ID to a vector of size d_model
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        The output should be scaled by sqrt(d_model) to maintain consistent
        magnitude regardless of embedding dimension.

        Args:
            x: Tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            Embedding tensor with shape (batch_size, seq_len, d_model).

        Hints:
            - Use self.embedding(x) to look up the embeddings.
            - Multiply by math.sqrt(self.d_model) to scale.
            - Why scale? Without scaling, the dot products in attention
              would shrink as d_model grows, making softmax less peaked.
        """
        raise NotImplementedError("TODO: implement forward")


class RotaryPositionalEmbeddingExercise(nn.Module):
    """Rotary Positional Embedding (RoPE) — exercise version.

    RoPE encodes positional information by rotating vectors.
    For a vector at position m, each pair of dimensions (2i, 2i+1)
    is rotated by angle m * theta_i, where theta_i = 1 / (base^(2i/d_model)).

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

        # TODO: Compute inverse frequencies
        # HINT: For each dimension pair i in [0, d_model/2):
        #       inv_freq[i] = 1 / (base^(2i/d_model))
        #       Use torch.arange(0, d_model, 2) for the indices,
        #       then compute base^(indices/d_model) and take reciprocal.
        #       Register as a buffer so it moves with the module.
        pass

    def _build_cache(self, seq_len: int, device: torch.device):
        """Precompute cosine and sine values for all positions and frequencies.

        Args:
            seq_len: Sequence length to compute for.
            device: Device to create tensors on.

        Returns:
            Tuple of (cos, sin) tensors, each with shape (seq_len, d_model).

        Hints:
            1. Create position indices: torch.arange(seq_len, device=device)
            2. Compute frequencies: outer product of positions and inv_freq
               freqs = torch.outer(positions, self.inv_freq)
               Shape: (seq_len, d_model/2)
            3. Duplicate to full d_model: torch.cat([freqs, freqs], dim=-1)
               Shape: (seq_len, d_model)
            4. Return (emb.cos(), emb.sin())
        """
        raise NotImplementedError("TODO: implement _build_cache")

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions by 90 degrees.

        For input [..., x1, x2], produces [..., -x2, x1].

        This is equivalent to multiplying each 2D vector by the
        90-degree rotation matrix [[0, -1], [1, 0]].

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Rotated tensor of same shape.

        Hints:
            1. Split x into two halves along the last dimension:
               x1 = x[..., :d_model//2]
               x2 = x[..., d_model//2:]
            2. Return torch.cat([-x2, x1], dim=-1)
        """
        raise NotImplementedError("TODO: implement _rotate_half")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor.

        The rotation formula is:
            x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)

        This applies a position-dependent rotation to each pair of
        dimensions, encoding the absolute position in the rotation angles.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Rotated tensor of same shape.

        Hints:
            1. Get seq_len from x.shape[1]
            2. Call _build_cache to get cos, sin
            3. Slice to seq_len and add batch dimension with unsqueeze(0)
            4. Apply the rotation formula:
               return x * cos + self._rotate_half(x) * sin
        """
        raise NotImplementedError("TODO: implement forward")
