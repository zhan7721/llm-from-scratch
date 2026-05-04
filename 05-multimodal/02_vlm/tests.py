"""Tests for the Visual Language Model (VLM) implementation."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from vlm import VisionProjector, VLM, VLMConfig


# ============================================================================
# Test VisionProjector
# ============================================================================


def test_projector_output_shape():
    """Projector should map (B, num_patches, vision_dim) to (B, num_patches, lm_dim)."""
    projector = VisionProjector(vision_dim=768, lm_dim=512)
    x = torch.randn(2, 197, 768)  # 197 patches (14x14 + 1 CLS)
    out = projector(x)
    assert out.shape == (2, 197, 512)


def test_projector_different_batch_sizes():
    """Projector should work with different batch sizes."""
    projector = VisionProjector(vision_dim=768, lm_dim=512)
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 197, 768)
        out = projector(x)
        assert out.shape == (batch_size, 197, 512)


def test_projector_different_num_patches():
    """Projector should work with different numbers of patches."""
    projector = VisionProjector(vision_dim=768, lm_dim=512)
    for num_patches in [64, 197, 256]:
        x = torch.randn(2, num_patches, 768)
        out = projector(x)
        assert out.shape == (2, num_patches, 512)


def test_projector_gradient_flow():
    """Gradients should flow through the projector."""
    projector = VisionProjector(vision_dim=768, lm_dim=512)
    x = torch.randn(2, 197, 768, requires_grad=True)
    out = projector(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_projector_parameter_count():
    """Projector should have a reasonable number of parameters."""
    projector = VisionProjector(vision_dim=768, lm_dim=512)
    total_params = sum(p.numel() for p in projector.parameters())
    assert total_params > 0
    # MLP: 768*512 + 512 + 512*512 + 512 = 393,216 + 512 + 262,144 + 512 = 656,384
    assert total_params < 1_000_000


def test_projector_architecture():
    """Projector should be an MLP with two linear layers and GELU activation."""
    projector = VisionProjector(vision_dim=768, lm_dim=512)
    # Check that it has the expected structure
    assert hasattr(projector, 'mlp')
    # The MLP should have at least 2 Linear layers
    linear_count = sum(1 for m in projector.mlp.modules() if isinstance(m, torch.nn.Linear))
    assert linear_count >= 2


# ============================================================================
# Test VLM
# ============================================================================


def test_vlm_forward_shape():
    """VLM forward should produce logits of shape (B, num_vision_tokens + seq_len, vocab_size)."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)

    # Create dummy inputs
    # For 32x32 image with 16x16 patches: (32/16)^2 = 4 patches + 1 CLS = 5 vision tokens
    images = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 256, (2, 10))

    logits = model(images, input_ids)
    expected_seq_len = 5 + 10  # vision tokens + text tokens
    assert logits.shape == (2, expected_seq_len, 256)


def test_vlm_vision_features_injected():
    """Vision features should be prepended to text embeddings."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)
    model.eval()

    images = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 256, (2, 10))

    # Get the combined embeddings before the language model
    with torch.no_grad():
        vision_features = model.vision_encoder(images)  # (B, num_vision_tokens, vision_dim)
        vision_embeddings = model.projector(vision_features)  # (B, num_vision_tokens, lm_dim)
        text_embeddings = model.language_model.token_emb(input_ids)  # (B, seq_len, lm_dim)

        # Vision embeddings should be prepended
        assert vision_embeddings.shape[0] == text_embeddings.shape[0]
        assert vision_embeddings.shape[2] == text_embeddings.shape[2]


def test_vlm_different_batch_sizes():
    """VLM should work with different batch sizes."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)

    for batch_size in [1, 2, 4]:
        images = torch.randn(batch_size, 3, 32, 32)
        input_ids = torch.randint(0, 256, (batch_size, 10))
        logits = model(images, input_ids)
        expected_seq_len = 5 + 10  # 5 vision tokens + 10 text tokens
        assert logits.shape == (batch_size, expected_seq_len, 256)


def test_vlm_different_text_lengths():
    """VLM should work with different text sequence lengths."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)

    for seq_len in [1, 5, 10, 50]:
        images = torch.randn(2, 3, 32, 32)
        input_ids = torch.randint(0, 256, (2, seq_len))
        logits = model(images, input_ids)
        expected_total = 5 + seq_len  # 5 vision tokens + text tokens
        assert logits.shape == (2, expected_total, 256)


def test_vlm_gradient_flow():
    """Gradients should flow through the entire VLM."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)

    images = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 256, (2, 10))

    logits = model(images, input_ids)
    loss = logits.sum()
    loss.backward()

    # Check that gradients flow to vision encoder
    for name, p in model.vision_encoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for vision_encoder.{name}"

    # Check that gradients flow to projector
    for name, p in model.projector.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for projector.{name}"

    # Check that gradients flow to language model
    for name, p in model.language_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for language_model.{name}"


def test_vlm_components_exist():
    """VLM should have vision_encoder, projector, and language_model components."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)

    assert hasattr(model, 'vision_encoder')
    assert hasattr(model, 'projector')
    assert hasattr(model, 'language_model')


def test_vlm_parameter_count():
    """VLM should have a reasonable number of parameters for a small config."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    # Should be under 2M for this small config
    assert total_params < 2_000_000


def test_vlm_output_not_all_same():
    """VLM output logits should not be identical across vocabulary dimension."""
    config = VLMConfig(
        vision_dim=64,
        lm_dim=64,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_layers=2,
        lm_max_seq_len=128,
        vision_image_size=32,
        vision_patch_size=16,
        vision_n_heads=4,
        vision_n_layers=2,
    )
    model = VLM(config)
    model.eval()

    images = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 256, (2, 10))

    with torch.no_grad():
        logits = model(images, input_ids)
        # Check that logits are not all the same value
        assert logits.std() > 0.01


# ============================================================================
# Test VLMConfig
# ============================================================================


def test_vlm_config_defaults():
    """VLMConfig should have sensible defaults."""
    config = VLMConfig()
    assert config.vision_dim > 0
    assert config.lm_dim > 0
    assert config.lm_vocab_size > 0
    assert config.lm_n_heads > 0
    assert config.lm_n_layers > 0
    assert config.lm_max_seq_len > 0


def test_vlm_config_custom_values():
    """VLMConfig should accept custom values."""
    config = VLMConfig(
        vision_dim=1024,
        lm_dim=256,
        lm_vocab_size=512,
        lm_n_heads=8,
        lm_n_layers=4,
        lm_max_seq_len=256,
    )
    assert config.vision_dim == 1024
    assert config.lm_dim == 256
    assert config.lm_vocab_size == 512
    assert config.lm_n_heads == 8
    assert config.lm_n_layers == 4
    assert config.lm_max_seq_len == 256
