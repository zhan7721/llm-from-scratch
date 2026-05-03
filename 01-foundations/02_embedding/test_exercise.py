"""Tests for the exercise version of Token Embedding and RoPE.

These tests verify that the exercise implementation matches the expected behavior.
"""

import torch
import pytest
from exercise import TokenEmbeddingExercise, RotaryPositionalEmbeddingExercise


# ============================================================================
# Token Embedding Exercise Tests
# ============================================================================


def test_token_exercise_shape():
    """Embedding output should have shape (batch, seq_len, d_model)."""
    emb = TokenEmbeddingExercise(vocab_size=1000, d_model=64)
    x = torch.randint(0, 1000, (2, 10))
    out = emb(x)
    assert out.shape == (2, 10, 64)


def test_token_exercise_gradients():
    """Embedding should support backpropagation."""
    emb = TokenEmbeddingExercise(vocab_size=1000, d_model=64)
    x = torch.randint(0, 1000, (2, 10))
    out = emb(x)
    out.sum().backward()
    assert emb.embedding.weight.grad is not None


def test_token_exercise_scaling():
    """Output should be scaled by sqrt(d_model)."""
    d_model = 64
    emb = TokenEmbeddingExercise(vocab_size=1000, d_model=d_model)
    x = torch.randint(0, 1000, (1, 1))
    out = emb(x)
    raw = emb.embedding(x)
    assert torch.allclose(out, raw * (d_model ** 0.5), atol=1e-6)


# ============================================================================
# Rotary Positional Embedding Exercise Tests
# ============================================================================


def test_rope_exercise_shape():
    """RoPE should preserve the input tensor shape."""
    rope = RotaryPositionalEmbeddingExercise(d_model=64, max_seq_len=128)
    x = torch.randn(2, 10, 64)
    out = rope(x)
    assert out.shape == x.shape


def test_rope_exercise_different_positions():
    """Different positions should produce different outputs."""
    rope = RotaryPositionalEmbeddingExercise(d_model=64, max_seq_len=128)
    x = torch.randn(1, 5, 64)
    out = rope(x)
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


def test_rope_exercise_preserves_norm():
    """RoPE rotation should preserve vector norms."""
    rope = RotaryPositionalEmbeddingExercise(d_model=64, max_seq_len=128)
    x = torch.randn(1, 5, 64)
    out = rope(x)
    for pos in range(5):
        in_norm = x[0, pos].norm().item()
        out_norm = out[0, pos].norm().item()
        assert abs(in_norm - out_norm) < 1e-4, f"Norm not preserved at position {pos}"


def test_rope_exercise_gradients():
    """RoPE should support backpropagation."""
    rope = RotaryPositionalEmbeddingExercise(d_model=64, max_seq_len=128)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = rope(x)
    out.sum().backward()
    assert x.grad is not None
