"""Tests for the Vision Transformer (ViT) implementation."""

import torch
import pytest
from vision_encoder import PatchEmbedding, VisionTransformer, ViTConfig


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------
def _small_config(**overrides) -> ViTConfig:
    """Create a small ViTConfig suitable for unit testing."""
    defaults = dict(
        image_size=32,
        patch_size=4,
        num_channels=3,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    defaults.update(overrides)
    return ViTConfig(**defaults)


def _dummy_images(config: ViTConfig, batch_size: int = 2) -> torch.Tensor:
    """Create dummy image tensors."""
    return torch.randn(
        batch_size, config.num_channels, config.image_size, config.image_size
    )


# ===========================================================================
# PatchEmbedding tests
# ===========================================================================

class TestPatchEmbedding:
    """Tests for the PatchEmbedding module."""

    def test_output_shape(self):
        """PatchEmbedding output shape should be (B, N, D)."""
        config = _small_config()
        pe = PatchEmbedding(
            config.image_size, config.patch_size,
            config.num_channels, config.embedding_dim,
        )
        x = _dummy_images(config, batch_size=2)
        out = pe(x)

        num_patches = (config.image_size // config.patch_size) ** 2
        assert out.shape == (2, num_patches, config.embedding_dim)

    def test_num_patches(self):
        """Number of patches should equal (H/P) * (W/P)."""
        config = _small_config(image_size=32, patch_size=4)
        pe = PatchEmbedding(32, 4, 3, 64)
        assert pe.num_patches == (32 // 4) * (32 // 4)  # 64

    def test_num_patches_different_sizes(self):
        """Patch count should scale with image and patch size."""
        pe1 = PatchEmbedding(64, 16, 3, 64)
        assert pe1.num_patches == (64 // 16) ** 2  # 16

        pe2 = PatchEmbedding(224, 16, 3, 768)
        assert pe2.num_patches == (224 // 16) ** 2  # 196

    def test_gradient_flows(self):
        """Gradients should flow through PatchEmbedding."""
        config = _small_config()
        pe = PatchEmbedding(
            config.image_size, config.patch_size,
            config.num_channels, config.embedding_dim,
        )
        x = _dummy_images(config, batch_size=1)
        out = pe(x)
        loss = out.sum()
        loss.backward()

        for name, p in pe.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_batch_independence(self):
        """Different batch elements should produce independent outputs."""
        config = _small_config()
        pe = PatchEmbedding(
            config.image_size, config.patch_size,
            config.num_channels, config.embedding_dim,
        )
        x = _dummy_images(config, batch_size=2)
        out = pe(x)

        # Outputs for different batch elements should differ
        assert not torch.allclose(out[0], out[1])


# ===========================================================================
# VisionTransformer tests
# ===========================================================================

class TestVisionTransformer:
    """Tests for the full VisionTransformer model."""

    def test_output_shape_with_head(self):
        """With classification head, output should be (B, num_classes)."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)
        x = _dummy_images(config, batch_size=2)
        out = model(x)
        assert out.shape == (2, 10)

    def test_output_shape_without_head(self):
        """Without classification head, output should be (B, N+1, D)."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=None)
        x = _dummy_images(config, batch_size=2)
        out = model(x)

        num_patches = (config.image_size // config.patch_size) ** 2
        assert out.shape == (2, num_patches + 1, config.embedding_dim)

    def test_cls_token_present(self):
        """CLS token should be a learnable parameter of shape (1, 1, D)."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)

        assert hasattr(model, "cls_token")
        assert isinstance(model.cls_token, torch.nn.Parameter)
        assert model.cls_token.shape == (1, 1, config.embedding_dim)

    def test_position_embedding_shape(self):
        """Position embedding should have shape (1, N+1, D)."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)

        num_patches = (config.image_size // config.patch_size) ** 2
        assert model.position_embedding.shape == (
            1, num_patches + 1, config.embedding_dim
        )

    def test_gradient_flows(self):
        """Gradients should flow through all parameters."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)
        x = _dummy_images(config, batch_size=2)

        logits = model(x)
        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_inference_mode(self):
        """Model in inference mode should produce correct output shape."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)
        model.eval()

        assert not model.training
        x = _dummy_images(config, batch_size=1)
        out = model(x)
        assert out.shape == (1, 10)

    def test_different_batch_sizes(self):
        """Model should work with different batch sizes."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)
        model.eval()

        for batch_size in [1, 2, 4]:
            x = _dummy_images(config, batch_size=batch_size)
            out = model(x)
            assert out.shape == (batch_size, 10)

    def test_different_num_classes(self):
        """Model should work with different num_classes values."""
        config = _small_config()

        for num_classes in [1, 10, 100, 1000]:
            model = VisionTransformer(config, num_classes=num_classes)
            x = _dummy_images(config, batch_size=1)
            out = model(x)
            assert out.shape == (1, num_classes)

    def test_config_defaults(self):
        """ViTConfig should have sensible defaults."""
        config = ViTConfig()
        assert config.image_size == 224
        assert config.patch_size == 16
        assert config.num_channels == 3
        assert config.embedding_dim == 768
        assert config.num_heads == 12
        assert config.num_layers == 12
        assert config.mlp_ratio == 4.0
        assert config.dropout == 0.0

    def test_num_transformer_blocks(self):
        """Model should have the correct number of transformer blocks."""
        config = _small_config(num_layers=4)
        model = VisionTransformer(config, num_classes=10)
        assert len(model.blocks) == 4

    def test_cls_token_in_output(self):
        """Without head, the CLS token (position 0) should be present in output."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=None)
        model.eval()

        x = _dummy_images(config, batch_size=1)
        out = model(x)

        # Output should have N+1 positions (CLS + patches)
        num_patches = (config.image_size // config.patch_size) ** 2
        assert out.shape[1] == num_patches + 1

    def test_output_not_all_same(self):
        """Output should not be all identical values."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)
        model.eval()

        x = _dummy_images(config, batch_size=1)
        out = model(x)
        assert out.std() > 0.01

    def test_image_size_must_be_divisible_by_patch_size(self):
        """Should raise an error if image_size is not divisible by patch_size."""
        with pytest.raises(AssertionError):
            PatchEmbedding(image_size=32, patch_size=5, num_channels=3, embedding_dim=64)

    def test_parameter_count_reasonable(self):
        """Small config should have a reasonable (non-zero, not huge) parameter count."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())

        assert total_params > 0
        assert total_params < 1_000_000  # small config should be under 1M

    def test_deterministic_inference(self):
        """In inference mode, the same input should produce the same output."""
        config = _small_config()
        model = VisionTransformer(config, num_classes=10)
        model.eval()

        x = _dummy_images(config, batch_size=1)

        out1 = model(x)
        out2 = model(x)

        assert torch.allclose(out1, out2)

    def test_different_image_sizes(self):
        """Model should work with different image sizes (config must match)."""
        for img_size, patch_size in [(16, 4), (32, 8), (64, 16)]:
            config = _small_config(image_size=img_size, patch_size=patch_size)
            model = VisionTransformer(config, num_classes=5)
            x = _dummy_images(config, batch_size=1)
            out = model(x)
            assert out.shape == (1, 5)
