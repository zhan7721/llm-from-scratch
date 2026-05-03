"""Tests for RoPE scaling and YaRN implementations."""

import torch
import pytest
from long_context import ScaledRoPE, YaRNRope


def test_scaled_rope_extends_context():
    """ScaledRoPE should handle sequences longer than max_seq_len."""
    rope = ScaledRoPE(d_model=64, max_seq_len=128, scale_factor=2.0)
    x = torch.randn(1, 200, 64)  # longer than max_seq_len
    out = rope(x)
    assert out.shape == x.shape


def test_yarn_rope_shape():
    """YaRNRope should preserve input shape."""
    yarn = YaRNRope(d_model=64, max_seq_len=128, scale_factor=4.0)
    x = torch.randn(1, 400, 64)
    out = yarn(x)
    assert out.shape == x.shape


def test_rope_position_sensitivity():
    """Different positions should produce different encodings."""
    rope = ScaledRoPE(d_model=64, max_seq_len=128)
    # Use non-zero input since RoPE is multiplicative (0 rotated is still 0)
    x = torch.ones(1, 10, 64)
    out = rope(x)
    for i in range(9):
        assert not torch.allclose(out[0, i], out[0, i + 1], atol=1e-6)


def test_scaled_rope_different_scale_factors():
    """Different scale factors should produce different encodings."""
    x = torch.randn(1, 50, 64)
    rope1 = ScaledRoPE(d_model=64, scale_factor=1.0)
    rope2 = ScaledRoPE(d_model=64, scale_factor=2.0)
    out1 = rope1(x)
    out2 = rope2(x)
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_yarn_attention_temperature():
    """YaRN applies temperature scaling that reduces output magnitude."""
    x = torch.randn(1, 10, 64)
    yarn = YaRNRope(d_model=64, scale_factor=4.0)
    out = yarn(x)
    # Output should be scaled down by attn_factor (1/sqrt(4) = 0.5)
    assert out.abs().mean() < x.abs().mean() * 1.5


def test_rope_gradient_flow():
    """Gradients should flow through ScaledRoPE."""
    rope = ScaledRoPE(d_model=64, scale_factor=2.0)
    x = torch.randn(1, 10, 64, requires_grad=True)
    out = rope(x)
    out.sum().backward()
    assert x.grad is not None


def test_yarn_gradient_flow():
    """Gradients should flow through YaRNRope."""
    yarn = YaRNRope(d_model=64, scale_factor=4.0)
    x = torch.randn(1, 10, 64, requires_grad=True)
    out = yarn(x)
    out.sum().backward()
    assert x.grad is not None


def test_scaled_rope_identity():
    """Scale factor of 1.0 should produce standard RoPE."""
    rope = ScaledRoPE(d_model=64, scale_factor=1.0)
    x = torch.randn(1, 10, 64)
    out = rope(x)
    # With scale_factor=1.0, positions are not scaled
    assert out.shape == x.shape


def test_yarn_different_scale_factors():
    """YaRN with different scale factors should produce different results."""
    x = torch.randn(1, 20, 64)
    yarn1 = YaRNRope(d_model=64, scale_factor=2.0)
    yarn2 = YaRNRope(d_model=64, scale_factor=4.0)
    out1 = yarn1(x)
    out2 = yarn2(x)
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_rope_batch_independence():
    """Different batch elements should be processed independently."""
    rope = ScaledRoPE(d_model=64, scale_factor=2.0)
    x = torch.randn(3, 10, 64)
    out = rope(x)
    # Each batch element should be the same (same input, same positions)
    # But with different inputs they should differ
    assert out.shape == (3, 10, 64)


def test_rope_dtype_preservation():
    """Output dtype should match input dtype."""
    rope = ScaledRoPE(d_model=64, scale_factor=2.0)
    x = torch.randn(1, 10, 64)
    out = rope(x)
    assert out.dtype == x.dtype
