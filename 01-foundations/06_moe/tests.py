"""Tests for Mixture of Experts (MoE) components."""

import torch
import pytest
from moe import MoELayer, TopKRouter, Expert


# ============================================================================
# Expert Tests
# ============================================================================


def test_expert_output_shape():
    """Expert output should have shape (..., d_model)."""
    expert = Expert(d_model=64, d_ff=128)
    x = torch.randn(5, 64)
    out = expert(x)
    assert out.shape == (5, 64)


def test_expert_batch_output_shape():
    """Expert should handle batched inputs."""
    expert = Expert(d_model=64, d_ff=128)
    x = torch.randn(3, 10, 64)
    out = expert(x)
    assert out.shape == (3, 10, 64)


def test_expert_gradient_flow():
    """Gradients should flow through all expert parameters."""
    expert = Expert(d_model=64, d_ff=128)
    x = torch.randn(5, 64, requires_grad=True)
    out = expert(x)
    out.sum().backward()

    assert x.grad is not None
    for name, param in expert.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# TopKRouter Tests
# ============================================================================


def test_router_output_shapes():
    """Router should return indices and weights of shape (B*T, top_k)."""
    router = TopKRouter(d_model=64, n_experts=4, top_k=2)
    x = torch.randn(2, 10, 64)
    indices, weights = router(x)
    assert indices.shape == (20, 2)
    assert weights.shape == (20, 2)


def test_router_indices_in_range():
    """Expert indices should be in [0, n_experts)."""
    router = TopKRouter(d_model=64, n_experts=4, top_k=2)
    x = torch.randn(1, 10, 64)
    indices, weights = router(x)
    assert (indices >= 0).all()
    assert (indices < 4).all()


def test_router_weights_sum_to_one():
    """Routing weights should sum to ~1 after softmax."""
    router = TopKRouter(d_model=64, n_experts=4, top_k=2)
    x = torch.randn(1, 10, 64)
    indices, weights = router(x)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0]), atol=1e-5)


def test_router_load_balancing():
    """Router should use at least 2 different experts for a random input."""
    router = TopKRouter(d_model=64, n_experts=4, top_k=2)
    x = torch.randn(1, 100, 64)
    indices, weights = router(x)
    unique_experts = torch.unique(indices)
    assert len(unique_experts) >= 2


def test_router_gradient_flow():
    """Gradients should flow through the router gate."""
    router = TopKRouter(d_model=64, n_experts=4, top_k=2)
    x = torch.randn(1, 10, 64, requires_grad=True)
    indices, weights = router(x)
    weights.sum().backward()
    assert x.grad is not None


# ============================================================================
# MoELayer Tests
# ============================================================================


def test_moe_output_shape():
    """MoE output should have the same shape as input."""
    moe = MoELayer(d_model=64, d_ff=128, n_experts=4, top_k=2)
    x = torch.randn(2, 10, 64)
    out = moe(x)
    assert out.shape == (2, 10, 64)


def test_moe_expert_count():
    """MoE should create the correct number of experts."""
    moe = MoELayer(d_model=64, d_ff=128, n_experts=8, top_k=2)
    assert len(moe.experts) == 8


def test_moe_gradient_flow():
    """Gradients should flow through the entire MoE layer."""
    moe = MoELayer(d_model=64, d_ff=128, n_experts=4, top_k=2)
    x = torch.randn(2, 10, 64, requires_grad=True)
    out = moe(x)
    out.sum().backward()
    assert x.grad is not None


def test_moe_all_params_have_gradients():
    """All MoE parameters should receive gradients."""
    moe = MoELayer(d_model=64, d_ff=128, n_experts=4, top_k=2)
    x = torch.randn(2, 10, 64)
    out = moe(x)
    out.sum().backward()

    for name, param in moe.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_moe_different_batch_sizes():
    """MoE should handle different batch sizes."""
    moe = MoELayer(d_model=64, d_ff=128, n_experts=4, top_k=2)
    for B, T in [(1, 5), (4, 20), (8, 1)]:
        x = torch.randn(B, T, 64)
        out = moe(x)
        assert out.shape == (B, T, 64)


def test_moe_topk1_reduces_to_single_expert():
    """With top_k=1, each token uses exactly one expert."""
    moe = MoELayer(d_model=64, d_ff=128, n_experts=4, top_k=1)
    x = torch.randn(2, 10, 64)
    out = moe(x)
    assert out.shape == (2, 10, 64)


def test_moe_deterministic_with_seed():
    """MoE should produce the same output for the same input and seed."""
    torch.manual_seed(42)
    moe = MoELayer(d_model=64, d_ff=128, n_experts=4, top_k=2)
    x = torch.randn(2, 10, 64)

    torch.manual_seed(0)
    out1 = moe(x)
    torch.manual_seed(0)
    out2 = moe(x)

    assert torch.allclose(out1, out2)
