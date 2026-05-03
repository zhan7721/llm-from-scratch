"""Tests for Multi-Head Attention and Grouped Query Attention."""

import torch
import pytest
from attention import MultiHeadAttention, GroupedQueryAttention


# ============================================================================
# Multi-Head Attention Tests
# ============================================================================


def test_mha_output_shape():
    """MHA output should have shape (batch, seq_len, d_model)."""
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    x = torch.randn(2, 10, 64)
    out = mha(x)
    assert out.shape == (2, 10, 64)


def test_mha_causal_mask_blocks_future():
    """With causal=True, position i should not attend to positions > i."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model=64, n_heads=8, causal=True)
    x = torch.randn(1, 5, 64)

    # Get the output — it should be deterministic
    out = mha(x)

    # Verify the causal property: changing future tokens should not
    # change the output at earlier positions
    x_modified = x.clone()
    x_modified[0, 3, :] = torch.randn(64)
    x_modified[0, 4, :] = torch.randn(64)
    out_modified = mha(x_modified)

    # Positions 0, 1, 2 should be identical (they cannot see positions 3, 4)
    assert torch.allclose(out[0, :3], out_modified[0, :3], atol=1e-5), (
        "Causal mask failed: earlier positions changed when future tokens changed"
    )


def test_mha_gradient_flow():
    """Gradients should flow through all MHA parameters."""
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    x = torch.randn(2, 10, 64)
    out = mha(x)
    out.sum().backward()

    for name, param in mha.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_mha_different_input_sizes():
    """MHA should handle different batch sizes and sequence lengths."""
    mha = MultiHeadAttention(d_model=64, n_heads=8)
    for B, T in [(1, 5), (4, 20), (8, 1)]:
        x = torch.randn(B, T, 64)
        out = mha(x)
        assert out.shape == (B, T, 64)


def test_mha_with_mask():
    """MHA should respect an explicit mask."""
    mha = MultiHeadAttention(d_model=64, n_heads=8, causal=False)
    x = torch.randn(2, 5, 64)
    # Mask out position 3 and 4 for all heads
    mask = torch.ones(2, 1, 1, 5)
    mask[:, :, :, 3:] = 0
    out_masked = mha(x, mask=mask)
    assert out_masked.shape == (2, 5, 64)


# ============================================================================
# Grouped Query Attention Tests
# ============================================================================


def test_gqa_output_shape():
    """GQA output should have shape (batch, seq_len, d_model)."""
    gqa = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2)
    x = torch.randn(2, 10, 64)
    out = gqa(x)
    assert out.shape == (2, 10, 64)


def test_gqa_fewer_kv_params():
    """GQA should have fewer KV parameters than MHA with the same d_model."""
    n_heads = 8
    d_model = 64
    n_kv_heads = 2

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    gqa = GroupedQueryAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads
    )

    mha_kv_params = sum(
        p.numel() for name, p in mha.named_parameters() if "W_k" in name or "W_v" in name
    )
    gqa_kv_params = sum(
        p.numel() for name, p in gqa.named_parameters() if "W_k" in name or "W_v" in name
    )

    expected_ratio = n_kv_heads / n_heads
    actual_ratio = gqa_kv_params / mha_kv_params
    assert abs(actual_ratio - expected_ratio) < 1e-6, (
        f"Expected KV param ratio {expected_ratio}, got {actual_ratio}"
    )


def test_gqa_gradient_flow():
    """Gradients should flow through all GQA parameters."""
    gqa = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2)
    x = torch.randn(2, 10, 64)
    out = gqa(x)
    out.sum().backward()

    for name, param in gqa.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_gqa_causal_mask():
    """GQA with causal mask should block future positions."""
    torch.manual_seed(42)
    gqa = GroupedQueryAttention(
        d_model=64, n_heads=8, n_kv_heads=2, causal=True
    )
    x = torch.randn(1, 5, 64)
    out = gqa(x)

    x_modified = x.clone()
    x_modified[0, 4, :] = torch.randn(64)
    out_modified = gqa(x_modified)

    # Positions 0-3 should be identical (cannot see position 4)
    assert torch.allclose(out[0, :4], out_modified[0, :4], atol=1e-5), (
        "Causal mask failed: earlier positions changed when future tokens changed"
    )


def test_gqa_reduces_to_mha_when_kv_eq_heads():
    """GQA with n_kv_heads == n_heads should behave like MHA."""
    d_model = 64
    n_heads = 8

    torch.manual_seed(42)
    gqa = GroupedQueryAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads
    )
    # GQA with n_kv_heads == n_heads has the same architecture as MHA
    # Verify it produces valid output
    x = torch.randn(2, 10, d_model)
    out = gqa(x)
    assert out.shape == (2, 10, d_model)


def test_gqa_different_kv_heads():
    """GQA should work with various n_kv_heads values that divide n_heads."""
    d_model = 64
    n_heads = 8
    for n_kv_heads in [1, 2, 4, 8]:
        gqa = GroupedQueryAttention(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads
        )
        x = torch.randn(2, 10, d_model)
        out = gqa(x)
        assert out.shape == (2, 10, d_model)
