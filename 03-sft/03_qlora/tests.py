import torch
import torch.nn as nn
import pytest
from qlora import NF4Quantizer, QuantizedLinear, QLoRALinear, apply_qlora, quantize_model_nf4


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(32, 32)
        self.W_k = nn.Linear(32, 32)
        self.other = nn.Linear(32, 16)

    def forward(self, x):
        return self.W_q(x)


def test_nf4_quantize_shape():
    quantizer = NF4Quantizer(block_size=16)
    tensor = torch.randn(32, 32)
    packed, scales = quantizer.quantize(tensor)
    assert packed.shape[0] > 0
    assert scales.shape[0] > 0


def test_nf4_roundtrip():
    quantizer = NF4Quantizer(block_size=16)
    tensor = torch.randn(32, 32)
    packed, scales = quantizer.quantize(tensor)
    dequant = quantizer.dequantize(packed, scales, tensor.shape)
    # Should be approximately equal (lossy quantization)
    assert dequant.shape == tensor.shape
    # Error should be small
    assert (tensor - dequant).abs().mean() < 0.1


def test_quantized_linear_shape():
    ql = QuantizedLinear(32, 32)
    ql.quantize_weights()
    x = torch.randn(2, 10, 32)
    out = ql(x)
    assert out.shape == (2, 10, 32)


def test_quantized_linear_memory():
    """Quantized weights should use less memory."""
    ql = QuantizedLinear(256, 256)
    original_size = ql.weight.numel() * 4  # float32
    ql.quantize_weights()
    # NF4 uses 4 bits per weight + scales
    quantized_size = ql._packed.numel() + ql._scales.numel() * 4
    assert quantized_size < original_size


def test_qlora_linear_shape():
    qlora = QLoRALinear(32, 32, rank=8)
    qlora.quantize_base()
    x = torch.randn(2, 10, 32)
    out = qlora(x)
    assert out.shape == (2, 10, 32)


def test_qlora_trainable():
    qlora = QLoRALinear(32, 32, rank=8)
    qlora.quantize_base()
    x = torch.randn(1, 5, 32)
    out = qlora(x)
    out.sum().backward()
    assert qlora.lora_A.grad is not None
    assert qlora.lora_B.grad is not None


def test_apply_qlora():
    model = DummyModel()
    model = apply_qlora(model, rank=8)
    assert isinstance(model.W_q, QLoRALinear)
    assert isinstance(model.W_k, QLoRALinear)
    assert isinstance(model.other, nn.Linear)


def test_qlora_forward():
    model = DummyModel()
    model = apply_qlora(model, rank=4)
    x = torch.randn(2, 5, 32)
    out = model(x)
    assert out.shape == (2, 5, 32)
