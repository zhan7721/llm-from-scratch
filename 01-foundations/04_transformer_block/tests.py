"""Tests for the Transformer Block implementation."""

import torch
import pytest
from transformer_block import TransformerBlock, SwiGLU, RMSNorm


def test_transformer_block_shape():
    """Output shape must match input shape."""
    block = TransformerBlock(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)


def test_transformer_block_residual():
    """Output should differ from input (not an identity function)."""
    block = TransformerBlock(d_model=64, n_heads=4)
    x = torch.randn(1, 5, 64)
    out = block(x)
    assert not torch.allclose(x, out, atol=1e-6)


def test_transformer_block_gradient():
    """Gradients should flow through the block."""
    block = TransformerBlock(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None


def test_transformer_block_deterministic():
    """Same input should produce same output (eval mode)."""
    block = TransformerBlock(d_model=64, n_heads=4)
    block.eval()
    x = torch.randn(1, 5, 64)
    out1 = block(x)
    out2 = block(x)
    assert torch.allclose(out1, out2)


def test_transformer_block_pre_norm():
    """Verify Pre-Norm structure: normalization applied before sub-layers."""
    block = TransformerBlock(d_model=64, n_heads=4)
    assert isinstance(block.attn_norm, RMSNorm)
    assert isinstance(block.ffn_norm, RMSNorm)


def test_transformer_block_causal():
    """Causal mode should be set on the attention layer."""
    block = TransformerBlock(d_model=64, n_heads=4, causal=True)
    assert block.attn.causal is True


def test_transformer_block_non_causal():
    """Non-causal mode should be set on the attention layer."""
    block = TransformerBlock(d_model=64, n_heads=4, causal=False)
    assert block.attn.causal is False


def test_swiglu_shape():
    """SwiGLU output should have same d_model as input."""
    ffn = SwiGLU(d_model=64, d_ff=128)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == (2, 10, 64)


def test_swiglu_default_hidden():
    """SwiGLU should compute a default d_ff when none is given."""
    ffn = SwiGLU(d_model=768)
    x = torch.randn(2, 10, 768)
    out = ffn(x)
    assert out.shape == (2, 10, 768)


def test_swiglu_gradient():
    """Gradients should flow through SwiGLU."""
    ffn = SwiGLU(d_model=64, d_ff=128)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = ffn(x)
    out.sum().backward()
    assert x.grad is not None


def test_rmsnorm_shape():
    """RMSNorm output shape should match input shape."""
    norm = RMSNorm(d_model=64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_rmsnorm_normalization():
    """RMSNorm should approximately produce unit RMS output."""
    norm = RMSNorm(d_model=64)
    x = torch.randn(1, 5, 64) * 100  # large values
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)


def test_rmsnorm_learnable_scale():
    """RMSNorm weight should be learnable."""
    norm = RMSNorm(d_model=64)
    assert norm.weight.requires_grad
    assert norm.weight.shape == (64,)


def test_rmsnorm_numerical_stability():
    """RMSNorm should handle near-zero inputs without NaN."""
    norm = RMSNorm(d_model=64)
    x = torch.randn(1, 5, 64) * 1e-8
    out = norm(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
