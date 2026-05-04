"""Tests for the DDPM diffusion implementation."""

import torch
import pytest
import torch.nn as nn
import math
from diffusion import (
    NoiseScheduler,
    SinusoidalTimeEmbedding,
    ResBlock,
    UNet,
    DDPMTrainer,
    ddpm_sample,
)


# ---------------------------------------------------------------------------
# Small config helpers for fast tests
# ---------------------------------------------------------------------------

def _small_scheduler(num_timesteps: int = 20) -> NoiseScheduler:
    """Create a small NoiseScheduler for fast testing."""
    return NoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=1e-4,
        beta_end=0.02,
    )


def _small_unet(
    in_channels: int = 3,
    base_channels: int = 16,
    time_emb_dim: int = 32,
) -> UNet:
    """Create a small UNet for fast testing."""
    return UNet(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_emb_dim=time_emb_dim,
    )


def _dummy_images(batch_size: int = 2, channels: int = 3, size: int = 16) -> torch.Tensor:
    """Create dummy image tensors."""
    return torch.randn(batch_size, channels, size, size)


# ===========================================================================
# NoiseScheduler tests
# ===========================================================================

class TestNoiseScheduler:
    """Tests for the NoiseScheduler class."""

    def test_init_creates_schedules(self):
        """Scheduler should create beta, alpha, and alpha_bar schedules."""
        scheduler = _small_scheduler(num_timesteps=100)
        assert scheduler.betas.shape == (100,)
        assert scheduler.alphas.shape == (100,)
        assert scheduler.alphas_cumprod.shape == (100,)

    def test_beta_linear_schedule(self):
        """Betas should increase linearly from beta_start to beta_end."""
        scheduler = NoiseScheduler(num_timesteps=100, beta_start=0.001, beta_end=0.1)
        assert torch.allclose(scheduler.betas[0], torch.tensor(0.001), atol=1e-6)
        assert torch.allclose(scheduler.betas[-1], torch.tensor(0.1), atol=1e-6)
        # Check linearity: differences should be constant
        diffs = scheduler.betas[1:] - scheduler.betas[:-1]
        assert torch.allclose(diffs, diffs[0], atol=1e-6)

    def test_alpha_values(self):
        """Alpha should equal 1 - beta."""
        scheduler = _small_scheduler()
        expected = 1.0 - scheduler.betas
        assert torch.allclose(scheduler.alphas, expected)

    def test_alphas_cumprod_decreasing(self):
        """Alpha_bar should be monotonically decreasing."""
        scheduler = _small_scheduler()
        diffs = scheduler.alphas_cumprod[1:] - scheduler.alphas_cumprod[:-1]
        assert (diffs < 0).all(), "alphas_cumprod should be strictly decreasing"

    def test_alphas_cumprod_range(self):
        """Alpha_bar should be in (0, 1]."""
        scheduler = _small_scheduler()
        assert (scheduler.alphas_cumprod > 0).all()
        assert (scheduler.alphas_cumprod <= 1.0).all()
        # First value should be close to 1 (small beta)
        assert scheduler.alphas_cumprod[0] > 0.99

    def test_add_noise_shape(self):
        """add_noise should return same shape as input."""
        scheduler = _small_scheduler()
        x_0 = _dummy_images(batch_size=2, channels=3, size=16)
        noise = torch.randn_like(x_0)
        timesteps = torch.tensor([0, 10])
        x_t = scheduler.add_noise(x_0, noise, timesteps)
        assert x_t.shape == x_0.shape

    def test_add_noise_timestep_zero(self):
        """At t=0, add_noise should return nearly clean images."""
        scheduler = _small_scheduler()
        x_0 = torch.ones(1, 3, 8, 8)
        noise = torch.randn_like(x_0)
        timesteps = torch.tensor([0])
        x_t = scheduler.add_noise(x_0, noise, timesteps)
        # sqrt(alpha_bar_0) ~ 1, so x_t ~ x_0
        assert torch.allclose(x_t, x_0, atol=0.1)

    def test_add_noise_timestep_max(self):
        """At t=T-1, add_noise should return nearly pure noise."""
        scheduler = _small_scheduler(num_timesteps=100)
        x_0 = torch.ones(1, 3, 8, 8)
        noise = torch.randn_like(x_0)
        timesteps = torch.tensor([99])
        x_t = scheduler.add_noise(x_0, noise, timesteps)
        # sqrt(alpha_bar_99) ~ 0, so x_t ~ noise
        # The signal should be mostly noise
        signal_ratio = (x_0 - x_t).pow(2).mean().sqrt()
        assert signal_ratio > 0.5, "At max timestep, output should be mostly noise"

    def test_add_noise_correct_amount(self):
        """add_noise should add the correct amount of noise per timestep."""
        scheduler = NoiseScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
        x_0 = torch.zeros(1, 1, 4, 4)
        noise = torch.ones_like(x_0)
        t = 500
        timesteps = torch.tensor([t])
        x_t = scheduler.add_noise(x_0, noise, timesteps)
        # x_t = sqrt(alpha_bar_t) * 0 + sqrt(1 - alpha_bar_t) * 1
        expected = math.sqrt(1.0 - scheduler.alphas_cumprod[t].item())
        assert torch.allclose(x_t, torch.tensor(expected), atol=1e-4)

    def test_add_noise_batch_independence(self):
        """Different batch elements should get different noise."""
        scheduler = _small_scheduler()
        x_0 = _dummy_images(batch_size=4)
        noise = torch.randn_like(x_0)
        timesteps = torch.tensor([5, 10, 15, 19])
        x_t = scheduler.add_noise(x_0, noise, timesteps)
        # Different timesteps should produce different outputs
        assert not torch.allclose(x_t[0], x_t[1])

    def test_step_shape(self):
        """scheduler.step should return same shape as input."""
        scheduler = _small_scheduler(num_timesteps=10)
        x_t = _dummy_images(batch_size=1, channels=3, size=8)
        model_output = torch.randn_like(x_t)
        x_prev = scheduler.step(model_output, timestep=5, sample=x_t)
        assert x_prev.shape == x_t.shape

    def test_step_timestep_zero(self):
        """At t=0, step should be deterministic (no noise added)."""
        scheduler = _small_scheduler(num_timesteps=10)
        x_t = _dummy_images(batch_size=1, channels=3, size=8)
        model_output = torch.randn_like(x_t)
        x_prev_1 = scheduler.step(model_output, timestep=0, sample=x_t)
        x_prev_2 = scheduler.step(model_output, timestep=0, sample=x_t)
        assert torch.allclose(x_prev_1, x_prev_2), "t=0 step should be deterministic"

    def test_step_timestep_nonzero(self):
        """At t>0, step should add noise (non-deterministic)."""
        scheduler = _small_scheduler(num_timesteps=10)
        x_t = _dummy_images(batch_size=1, channels=3, size=8)
        model_output = torch.randn_like(x_t)
        x_prev_1 = scheduler.step(model_output, timestep=5, sample=x_t)
        x_prev_2 = scheduler.step(model_output, timestep=5, sample=x_t)
        # With very high probability, these should differ
        assert not torch.allclose(x_prev_1, x_prev_2, atol=1e-6)

    def test_posterior_variance_range(self):
        """Posterior variance should be non-negative."""
        scheduler = _small_scheduler()
        assert (scheduler.posterior_variance >= 0).all()

    def test_precomputed_quantities_device_transfer(self):
        """Precomputed quantities should transfer to device correctly."""
        scheduler = _small_scheduler()
        x_0 = _dummy_images()
        noise = torch.randn_like(x_0)
        timesteps = torch.tensor([0, 5])
        # Should work without errors
        x_t = scheduler.add_noise(x_0, noise, timesteps)
        assert x_t.device == x_0.device


# ===========================================================================
# SinusoidalTimeEmbedding tests
# ===========================================================================

class TestSinusoidalTimeEmbedding:
    """Tests for SinusoidalTimeEmbedding."""

    def test_output_shape(self):
        """Output shape should be (B, dim)."""
        emb = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([0, 5, 10])
        out = emb(t)
        assert out.shape == (3, 64)

    def test_output_range(self):
        """Output values should be in [-1, 1] (sin/cos range)."""
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.arange(100)
        out = emb(t)
        assert (out >= -1.0).all()
        assert (out <= 1.0).all()

    def test_different_timesteps_different_embeddings(self):
        """Different timesteps should produce different embeddings."""
        emb = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([0, 1, 2, 3])
        out = emb(t)
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                assert not torch.allclose(out[i], out[j])

    def test_deterministic(self):
        """Same timestep should always produce same embedding."""
        emb = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([42])
        out1 = emb(t)
        out2 = emb(t)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self):
        """Gradients should flow through the embedding."""
        emb = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([0, 1, 2], dtype=torch.float32, requires_grad=True)
        out = emb(t)
        loss = out.sum()
        loss.backward()
        assert t.grad is not None


# ===========================================================================
# ResBlock tests
# ===========================================================================

class TestResBlock:
    """Tests for the ResBlock."""

    def test_output_shape_same_channels(self):
        """With same in/out channels, output shape matches input."""
        block = ResBlock(in_channels=16, out_channels=16, time_emb_dim=32)
        x = torch.randn(2, 16, 8, 8)
        t_emb = torch.randn(2, 32)
        out = block(x, t_emb)
        assert out.shape == (2, 16, 8, 8)

    def test_output_shape_different_channels(self):
        """With different in/out channels, output has out_channels."""
        block = ResBlock(in_channels=16, out_channels=32, time_emb_dim=32)
        x = torch.randn(2, 16, 8, 8)
        t_emb = torch.randn(2, 32)
        out = block(x, t_emb)
        assert out.shape == (2, 32, 8, 8)

    def test_residual_connection(self):
        """Block should have a residual (shortcut) connection."""
        block = ResBlock(in_channels=16, out_channels=16, time_emb_dim=32)
        assert isinstance(block.shortcut, nn.Identity)

    def test_gradient_flows(self):
        """Gradients should flow through the block."""
        block = ResBlock(in_channels=16, out_channels=16, time_emb_dim=32)
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        t_emb = torch.randn(1, 32)
        out = block(x, t_emb)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ===========================================================================
# UNet tests
# ===========================================================================

class TestUNet:
    """Tests for the UNet model."""

    def test_output_shape(self):
        """UNet output should match input shape."""
        model = _small_unet(in_channels=3, base_channels=16)
        x = torch.randn(2, 3, 16, 16)
        t = torch.tensor([0, 5])
        out = model(x, t)
        assert out.shape == (2, 3, 16, 16)

    def test_different_batch_sizes(self):
        """Model should work with different batch sizes."""
        model = _small_unet()
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 16, 16)
            t = torch.zeros(batch_size, dtype=torch.long)
            out = model(x, t)
            assert out.shape == (batch_size, 3, 16, 16)

    def test_different_timesteps(self):
        """Model should handle different timesteps."""
        model = _small_unet()
        x = torch.randn(1, 3, 16, 16)
        for t_val in [0, 5, 10, 19]:
            t = torch.tensor([t_val])
            out = model(x, t)
            assert out.shape == (1, 3, 16, 16)

    def test_gradient_flows(self):
        """Gradients should flow through all parameters."""
        model = _small_unet()
        x = torch.randn(1, 3, 16, 16)
        t = torch.tensor([5])
        out = model(x, t)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self):
        """Model should work in eval mode."""
        model = _small_unet()
        model.eval()
        x = torch.randn(1, 3, 16, 16)
        t = torch.tensor([5])
        out = model(x, t)
        assert out.shape == (1, 3, 16, 16)

    def test_in_channels_1(self):
        """Model should work with 1 channel (grayscale)."""
        model = _small_unet(in_channels=1, base_channels=16)
        x = torch.randn(1, 1, 16, 16)
        t = torch.tensor([0])
        out = model(x, t)
        assert out.shape == (1, 1, 16, 16)

    def test_time_embedding_present(self):
        """Model should have a time embedding module."""
        model = _small_unet()
        assert hasattr(model, "time_embedding")

    def test_skip_connections(self):
        """Model should have down_blocks and up_blocks for skip connections."""
        model = _small_unet()
        assert hasattr(model, "down_blocks")
        assert hasattr(model, "up_blocks")
        assert len(model.down_blocks) > 0
        assert len(model.up_blocks) > 0

    def test_deterministic_in_eval(self):
        """In eval mode with no randomness, same input gives same output."""
        model = _small_unet()
        model.eval()
        x = torch.randn(1, 3, 16, 16)
        t = torch.tensor([5])
        out1 = model(x, t)
        out2 = model(x, t)
        assert torch.allclose(out1, out2)

    def test_parameter_count_reasonable(self):
        """Small model should have reasonable parameter count."""
        model = _small_unet()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 1_000_000  # small model under 1M params


# ===========================================================================
# DDPMTrainer tests
# ===========================================================================

class TestDDPMTrainer:
    """Tests for the DDPMTrainer class."""

    def test_train_step_returns_scalar(self):
        """train_step should return a scalar loss value."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = DDPMTrainer(model, scheduler, optimizer, device="cpu")

        x_0 = _dummy_images(batch_size=2, channels=3, size=16)
        loss = trainer.train_step(x_0)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_updates_parameters(self):
        """train_step should update model parameters."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = DDPMTrainer(model, scheduler, optimizer, device="cpu")

        x_0 = _dummy_images(batch_size=2, channels=3, size=16)

        # Save initial parameters
        initial_params = {
            name: p.clone() for name, p in model.named_parameters()
        }

        # Train step
        trainer.train_step(x_0)

        # At least some parameters should have changed
        changed = False
        for name, p in model.named_parameters():
            if not torch.allclose(p, initial_params[name]):
                changed = True
                break
        assert changed, "Parameters should change after train_step"

    def test_train_step_multiple_calls(self):
        """Multiple train_step calls should decrease loss over time."""
        torch.manual_seed(42)
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = DDPMTrainer(model, scheduler, optimizer, device="cpu")

        x_0 = _dummy_images(batch_size=4, channels=3, size=16)
        losses = []
        for _ in range(20):
            loss = trainer.train_step(x_0)
            losses.append(loss)

        # Loss should generally decrease (check last 5 vs first 5)
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg, f"Loss should decrease: early={early_avg:.4f}, late={late_avg:.4f}"

    def test_train_step_model_in_train_mode(self):
        """Model should be in train mode during train_step."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = DDPMTrainer(model, scheduler, optimizer, device="cpu")

        model.eval()  # Start in eval mode
        x_0 = _dummy_images(batch_size=2, channels=3, size=16)
        trainer.train_step(x_0)
        assert model.training, "Model should be in train mode after train_step"


# ===========================================================================
# ddpm_sample tests
# ===========================================================================

class TestDdpmSample:
    """Tests for the ddpm_sample function."""

    def test_output_shape(self):
        """Sampling should produce correct output shape."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)  # small T for speed
        shape = (2, 3, 16, 16)
        samples = ddpm_sample(model, scheduler, shape, device="cpu")
        assert samples.shape == shape

    def test_output_not_all_zeros(self):
        """Samples should not be all zeros."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        shape = (1, 3, 16, 16)
        samples = ddpm_sample(model, scheduler, shape, device="cpu")
        assert samples.abs().sum() > 0, "Samples should not be all zeros"

    def test_output_not_all_same(self):
        """Samples should have variation (not all same value)."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        shape = (1, 3, 16, 16)
        samples = ddpm_sample(model, scheduler, shape, device="cpu")
        assert samples.std() > 0.01, "Samples should have variation"

    def test_batch_independence(self):
        """Different batch elements should produce different samples."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        shape = (2, 3, 16, 16)
        samples = ddpm_sample(model, scheduler, shape, device="cpu")
        assert not torch.allclose(samples[0], samples[1])

    def test_model_in_eval_mode(self):
        """Model should be in eval mode during sampling."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        shape = (1, 3, 16, 16)
        ddpm_sample(model, scheduler, shape, device="cpu")
        assert not model.training, "Model should be in eval mode after sampling"

    def test_different_shapes(self):
        """Sampling should work with different spatial sizes."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        # Note: spatial size must be divisible by 4 (two downsamples)
        for size in [8, 16, 32]:
            shape = (1, 3, size, size)
            samples = ddpm_sample(model, scheduler, shape, device="cpu")
            assert samples.shape == shape

    def test_single_channel(self):
        """Sampling should work with single channel (grayscale)."""
        model = _small_unet(in_channels=1, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        shape = (1, 1, 16, 16)
        samples = ddpm_sample(model, scheduler, shape, device="cpu")
        assert samples.shape == shape

    def test_deterministic_with_seed(self):
        """With same random seed, sampling should produce same results."""
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        shape = (1, 3, 16, 16)

        torch.manual_seed(42)
        samples1 = ddpm_sample(model, scheduler, shape, device="cpu")

        torch.manual_seed(42)
        samples2 = ddpm_sample(model, scheduler, shape, device="cpu")

        assert torch.allclose(samples1, samples2)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Full training + sampling pipeline should work end-to-end."""
        torch.manual_seed(42)

        # Create small model and scheduler
        model = _small_unet(in_channels=3, base_channels=16)
        scheduler = _small_scheduler(num_timesteps=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = DDPMTrainer(model, scheduler, optimizer, device="cpu")

        # Train for a few steps
        x_0 = _dummy_images(batch_size=2, channels=3, size=16)
        for _ in range(5):
            trainer.train_step(x_0)

        # Sample
        shape = (1, 3, 16, 16)
        samples = ddpm_sample(model, scheduler, shape, device="cpu")
        assert samples.shape == shape

    def test_scheduler_device_consistency(self):
        """Scheduler tensors should work on the same device as model."""
        scheduler = _small_scheduler()
        x_0 = _dummy_images()
        noise = torch.randn_like(x_0)
        timesteps = torch.tensor([0, 5])
        x_t = scheduler.add_noise(x_0, noise, timesteps)
        assert x_t.device == x_0.device
