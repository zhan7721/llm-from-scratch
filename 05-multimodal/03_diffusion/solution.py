"""Reference solution for the DDPM exercise."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class NoiseSchedulerSolution:
    """Manages the forward diffusion (noising) and reverse (denoising) schedule."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Solution 1: Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Solution 2: alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # Solution 3: alpha_bar_t = cumulative product
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Solution 4: Precompute signal and noise coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Solution 5: Precompute reverse process quantities
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.beta_over_sqrt_one_minus_alpha_bar = self.betas / self.sqrt_one_minus_alphas_cumprod

        # Solution 6: Posterior variance
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to clean images (forward diffusion process)."""
        # Solution 7: Look up coefficients for each timestep
        sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(x_0.device)[timesteps]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)[timesteps]

        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)

        # Solution 8: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one reverse diffusion step (denoising)."""
        t = timestep

        # Solution 9: Predicted mean
        pred_mean = (
            (1.0 / self.sqrt_alphas[t]) *
            (sample - self.beta_over_sqrt_one_minus_alpha_bar[t] * model_output)
        )

        # Solution 10: Add noise for t > 0
        if t == 0:
            return pred_mean
        else:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(self.posterior_variance[t])
            return pred_mean + variance * noise


class SinusoidalTimeEmbeddingSolution(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for timesteps."""
        # Solution 11: Sinusoidal embedding
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlockSolution(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
    ):
        super().__init__()
        # Solution 12: GroupNorm + Conv2d layers
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Solution 13: Time embedding MLP
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Solution 14: Shortcut connection
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding."""
        # Solution 15: Forward pass
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class DDPMTrainerSolution:
    """Training loop for DDPM with noise prediction loss."""

    def __init__(
        self,
        model,
        scheduler,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device

    def train_step(self, x_0: torch.Tensor) -> float:
        """Perform one training step."""
        # Solution 16: DDPM training step
        self.model.train()
        x_0 = x_0.to(self.device)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Create noisy image
        x_t = self.scheduler.add_noise(x_0, noise, t)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
