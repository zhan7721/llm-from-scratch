import torch
import torch.nn as nn
import pytest
from lora import LoRALinear, apply_lora, get_lora_parameters, count_lora_params, merge_lora, unmerge_lora


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(64, 64)
        self.W_k = nn.Linear(64, 64)
        self.W_v = nn.Linear(64, 64)
        self.other = nn.Linear(64, 32)

    def forward(self, x):
        return self.W_v(x)  # simplified


def test_lora_linear_shape():
    lora = LoRALinear(64, 64, rank=8)
    x = torch.randn(2, 10, 64)
    out = lora(x)
    assert out.shape == (2, 10, 64)


def test_lora_rank_effect():
    """Higher rank should have more parameters."""
    lora_low = LoRALinear(64, 64, rank=4)
    lora_high = LoRALinear(64, 64, rank=16)
    params_low = sum(p.numel() for p in lora_low.parameters() if p.requires_grad)
    params_high = sum(p.numel() for p in lora_high.parameters() if p.requires_grad)
    assert params_high > params_low


def test_lora_frozen_original():
    """Original weights should be frozen."""
    lora = LoRALinear(64, 64, rank=8)
    assert not lora.linear.weight.requires_grad


def test_lora_trainable_params():
    """LoRA A and B should be trainable."""
    lora = LoRALinear(64, 64, rank=8)
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_lora_gradient_flow():
    lora = LoRALinear(64, 64, rank=8)
    x = torch.randn(2, 10, 64)
    out = lora(x)
    out.sum().backward()
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_apply_lora():
    model = DummyModel()
    model = apply_lora(model, rank=8)
    # W_q, W_k, W_v should be replaced
    assert isinstance(model.W_q, LoRALinear)
    assert isinstance(model.W_k, LoRALinear)
    assert isinstance(model.W_v, LoRALinear)
    # other should not be replaced
    assert isinstance(model.other, nn.Linear)


def test_lora_params_count():
    model = DummyModel()
    model = apply_lora(model, rank=8)
    lora_params, total_params = count_lora_params(model)
    assert lora_params > 0
    assert lora_params < total_params


def test_merge_unmerge():
    model = DummyModel()
    model = apply_lora(model, rank=8)

    # Set non-zero LoRA weights (B starts at zero, so simulate training)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                nn.init.normal_(module.lora_B, mean=0.0, std=0.01)

    x = torch.randn(1, 5, 64)
    out_before = model.W_q(x)

    merge_lora(model)
    out_merged = model.W_q(x)

    unmerge_lora(model)
    out_unmerged = model.W_q(x)

    # Before and after unmerge should be the same
    assert torch.allclose(out_before, out_unmerged, atol=1e-5)
    # Merged should be different (LoRA was added)
    assert not torch.allclose(out_before, out_merged, atol=1e-5)


def test_lora_scaling():
    """Different alpha should produce different outputs."""
    lora1 = LoRALinear(64, 64, rank=8, alpha=8)
    lora2 = LoRALinear(64, 64, rank=8, alpha=32)

    # Copy same LoRA weights
    lora2.lora_A.data = lora1.lora_A.data.clone()
    lora2.lora_B.data = lora1.lora_B.data.clone()

    x = torch.randn(1, 5, 64)
    out1 = lora1(x)
    out2 = lora2(x)
    assert not torch.allclose(out1, out2, atol=1e-5)


def test_lora_zero_init():
    """At initialization, LoRA should output the same as original (B is zero)."""
    lora = LoRALinear(64, 64, rank=8)
    x = torch.randn(1, 5, 64)
    orig_out = lora.linear(x)
    lora_out = lora(x)
    assert torch.allclose(orig_out, lora_out, atol=1e-5)
