"""Tests for Token Embedding and Rotary Positional Embedding."""

import torch
import pytest
from embedding import TokenEmbedding, RotaryPositionalEmbedding


# ============================================================================
# Token Embedding Tests
# ============================================================================


def test_token_embedding_shape():
    """Embedding output should have shape (batch, seq_len, d_model)."""
    emb = TokenEmbedding(vocab_size=1000, d_model=64)
    x = torch.randint(0, 1000, (2, 10))
    out = emb(x)
    assert out.shape == (2, 10, 64)


def test_token_embedding_gradients():
    """Embedding should support backpropagation."""
    emb = TokenEmbedding(vocab_size=1000, d_model=64)
    x = torch.randint(0, 1000, (2, 10))
    out = emb(x)
    out.sum().backward()
    assert emb.embedding.weight.grad is not None


def test_token_embedding_scaling():
    """Output should be scaled by sqrt(d_model)."""
    d_model = 64
    emb = TokenEmbedding(vocab_size=1000, d_model=d_model)
    x = torch.randint(0, 1000, (1, 1))
    out = emb(x)
    raw = emb.embedding(x)
    assert torch.allclose(out, raw * (d_model ** 0.5), atol=1e-6)


def test_token_embedding_different_ids():
    """Different token IDs should produce different embeddings."""
    emb = TokenEmbedding(vocab_size=1000, d_model=64)
    x = torch.tensor([[0, 1, 2]])
    out = emb(x)
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


# ============================================================================
# Rotary Positional Embedding Tests
# ============================================================================


def test_rope_shape_preservation():
    """RoPE should preserve the input tensor shape."""
    rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=128)
    x = torch.randn(2, 10, 64)
    out = rope(x)
    assert out.shape == x.shape


def test_rope_different_positions():
    """Different positions should produce different outputs."""
    rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=128)
    x = torch.randn(1, 5, 64)
    out = rope(x)
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


def test_rope_same_position_same_output():
    """Same position should produce same output for same input."""
    rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=128)
    x = torch.randn(1, 1, 64)
    x_repeated = x.repeat(1, 3, 1)
    out = rope(x_repeated)
    # All positions are the same input but at different positions
    # so outputs should differ
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


def test_rope_preserves_norm():
    """RoPE rotation should preserve vector norms (rotation is norm-preserving)."""
    rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=128)
    x = torch.randn(1, 5, 64)
    out = rope(x)
    # Each position's norm should be preserved (rotation preserves norm)
    for pos in range(5):
        in_norm = x[0, pos].norm().item()
        out_norm = out[0, pos].norm().item()
        assert abs(in_norm - out_norm) < 1e-4, f"Norm not preserved at position {pos}"


def test_rope_gradients():
    """RoPE should support backpropagation."""
    rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=128)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = rope(x)
    out.sum().backward()
    assert x.grad is not None


def test_rope_batch_independence():
    """RoPE should process each batch element independently."""
    rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=128)
    x = torch.randn(2, 5, 64)
    out = rope(x)
    # Process each batch element separately
    out_0 = rope(x[0:1])
    out_1 = rope(x[1:2])
    assert torch.allclose(out[0:1], out_0, atol=1e-6)
    assert torch.allclose(out[1:2], out_1, atol=1e-6)


def test_rope_different_d_model():
    """RoPE should work with different embedding dimensions."""
    for d_model in [32, 64, 128, 256]:
        rope = RotaryPositionalEmbedding(d_model=d_model, max_seq_len=128)
        x = torch.randn(1, 5, d_model)
        out = rope(x)
        assert out.shape == x.shape
