"""Denoising Diffusion Probabilistic Models (DDPM) implementation.

This module implements the core components of DDPM (Ho et al., 2020):
- NoiseScheduler: Manages the forward (noising) and reverse (denoising) process
- UNet: Simplified U-Net architecture for noise prediction
- DDPMTrainer: Training loop with noise prediction loss
- ddpm_sample: Iterative denoising sampling loop

DDPM generates images by learning to reverse a gradual noising process.
The forward process progressively adds Gaussian noise to clean images over T
timesteps. The reverse process learns to denoise step-by-step, recovering
clean images from pure noise.

Key equations:
    Forward:  q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Reverse:  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) + sigma_t * z

Design notes:
- This module is self-contained (no imports from other chapters).
- Uses standard DDPM conventions: linear beta schedule, sinusoidal time embedding.
- Small model friendly for educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class NoiseScheduler:
    """Manages the forward diffusion (noising) and reverse (denoising) schedule.

    The forward process gradually adds Gaussian noise to clean images according
    to a linear schedule of beta values. This class precomputes the cumulative
    products (alpha_bar) needed for efficient noise addition and removal.

    Args:
        num_timesteps: Number of diffusion steps T.
        beta_start: Starting value of beta (noise schedule).
        beta_end: Ending value of beta (noise schedule).
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear beta schedule: beta_t increases linearly from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # alpha_bar_t = prod(alpha_1, ..., alpha_t) = cumulative product
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute useful quantities for the forward process
        # sqrt(alpha_bar_t) -- signal coefficient
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # sqrt(1 - alpha_bar_t) -- noise coefficient
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Precompute quantities for the reverse process
        # sqrt(alpha_t) for the reverse step
        self.sqrt_alphas = torch.sqrt(self.alphas)
        # beta_t / sqrt(1 - alpha_bar_t) for the predicted mean
        self.beta_over_sqrt_one_minus_alpha_bar = self.betas / self.sqrt_one_minus_alphas_cumprod

        # Posterior variance: beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        # For t=0, this is 0 (no noise added at the final step)
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
        """Add noise to clean images (forward diffusion process).

        Implements: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: Clean images of shape (B, C, H, W).
            noise: Gaussian noise of same shape as x_0.
            timesteps: Timestep indices of shape (B,) with values in [0, T).

        Returns:
            Noisy images x_t of same shape as x_0.
        """
        sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(x_0.device)[timesteps]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)[timesteps]

        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one reverse diffusion step (denoising).

        Implements the reverse process:
            x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z

        For t=0, no noise is added (deterministic step).

        Args:
            model_output: Predicted noise epsilon_theta(x_t, t).
            timestep: Current timestep t (scalar).
            sample: Current noisy sample x_t.

        Returns:
            Denoised sample x_{t-1}.
        """
        t = timestep

        # Predicted mean: mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
        pred_mean = (
            (1.0 / self.sqrt_alphas[t]) *
            (sample - self.beta_over_sqrt_one_minus_alpha_bar[t] * model_output)
        )

        if t == 0:
            # Final step: no noise added
            return pred_mean
        else:
            # Add noise: x_{t-1} = mu_theta + sigma_t * z
            noise = torch.randn_like(sample)
            variance = torch.sqrt(self.posterior_variance[t])
            return pred_mean + variance * noise


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Similar to Transformer positional encodings, this embeds scalar timestep
    indices into dense vectors using sinusoidal functions of different frequencies.

    Args:
        dim: Dimension of the embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for timesteps.

        Args:
            t: Timestep tensor of shape (B,).

        Returns:
            Embeddings of shape (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        # log(10000) / (half_dim - 1) gives frequency scaling
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # (B,) * (half_dim,) -> (B, half_dim) via broadcasting
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        # Concatenate sin and cos: (B, dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time embedding injection.

    Each block applies two convolutions with GroupNorm and SiLU activation,
    plus a time embedding that is added after the first convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_emb_dim: Dimension of the time embedding.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Shortcut connection if channels change
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding.

        Args:
            x: Input tensor of shape (B, C, H, W).
            t_emb: Time embedding of shape (B, time_emb_dim).

        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time embedding: (B, out_channels) -> (B, out_channels, 1, 1)
        h = h + self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class UNet(nn.Module):
    """Simplified U-Net for noise prediction in DDPM.

    A U-Net architecture with:
    - Encoder: downsample blocks with skip connections
    - Bottleneck: middle block
    - Decoder: upsample blocks with skip connections
    - Time embedding: sinusoidal embedding for timestep conditioning

    The model predicts the noise epsilon added to a noisy image x_t at
    timestep t. This is used in the DDPM training objective:
        L = MSE(epsilon, epsilon_theta(x_t, t))

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB).
        base_channels: Base channel count (doubled at each downsample).
        channel_mults: Multipliers for channels at each resolution level.
        num_res_blocks: Number of residual blocks per resolution level.
        time_emb_dim: Dimension of the sinusoidal time embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding network
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Compute channel list for each resolution level
        channels = [base_channels]
        for mult in channel_mults:
            channels.append(base_channels * mult)

        # Encoder (downsampling) path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        for i in range(len(channel_mults)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            block = nn.ModuleList([
                ResBlock(in_ch, out_ch, time_emb_dim)
                for _ in range(num_res_blocks)
            ])
            self.down_blocks.append(block)
            # Downsample with stride-2 convolution (except at last level)
            if i < len(channel_mults) - 1:
                self.down_samples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))

        # Bottleneck
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_emb_dim)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim)

        # Decoder (upsampling) path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i in reversed(range(len(channel_mults))):
            in_ch = channels[i + 1]
            out_ch = channels[i]
            # After concatenation with skip connection, input channels double
            block = nn.ModuleList([
                ResBlock(in_ch * 2 if j == 0 else out_ch, out_ch, time_emb_dim)
                for j in range(num_res_blocks)
            ])
            self.up_blocks.append(block)
            # Upsample (except at first decoder level which matches bottleneck)
            if i > 0:
                self.up_samples.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))

        # Final output
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy image and timestep.

        Args:
            x: Noisy image of shape (B, C, H, W).
            t: Timestep indices of shape (B,).

        Returns:
            Predicted noise of same shape as x.
        """
        # Time embedding: (B,) -> (B, time_emb_dim)
        t_emb = self.time_embedding(t)

        # Initial convolution
        h = self.init_conv(x)

        # Encoder: collect skip connections
        skips = []
        for i, (block, down_sample) in enumerate(zip(self.down_blocks, self.down_samples)):
            for res_block in block:
                h = res_block(h, t_emb)
            skips.append(h)
            h = down_sample(h)
        # Last encoder level (no downsampling after it)
        for res_block in self.down_blocks[-1]:
            h = res_block(h, t_emb)
        skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Decoder: use skip connections
        for i, (block, up_sample) in enumerate(zip(self.up_blocks, self.up_samples)):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for res_block in block:
                h = res_block(h, t_emb)
            h = up_sample(h)
        # Last decoder level
        skip = skips.pop()
        h = torch.cat([h, skip], dim=1)
        for res_block in self.up_blocks[-1]:
            h = res_block(h, t_emb)

        # Final output
        h = self.final_conv(F.silu(self.final_norm(h)))
        return h


class DDPMTrainer:
    """Training loop for DDPM with noise prediction loss.

    Implements the simplified DDPM training objective:
        1. Sample clean image x_0
        2. Sample random timestep t ~ Uniform({0, ..., T-1})
        3. Sample noise epsilon ~ N(0, I)
        4. Create noisy image x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        5. Predict noise: epsilon_theta(x_t, t)
        6. Loss = MSE(epsilon, epsilon_theta)

    Args:
        model: UNet model for noise prediction.
        scheduler: NoiseScheduler for forward/reverse process.
        optimizer: PyTorch optimizer.
        device: Device to run on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: UNet,
        scheduler: NoiseScheduler,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device

    def train_step(self, x_0: torch.Tensor) -> float:
        """Perform one training step.

        Args:
            x_0: Clean images of shape (B, C, H, W).

        Returns:
            Loss value (scalar float).
        """
        self.model.train()
        x_0 = x_0.to(self.device)
        batch_size = x_0.shape[0]

        # Sample random timesteps: t ~ Uniform({0, ..., T-1})
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)

        # Sample noise: epsilon ~ N(0, I)
        noise = torch.randn_like(x_0)

        # Create noisy image: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t = self.scheduler.add_noise(x_0, noise, t)

        # Predict noise: epsilon_theta(x_t, t)
        predicted_noise = self.model(x_t, t)

        # Loss = MSE(epsilon, epsilon_theta)
        loss = F.mse_loss(predicted_noise, noise)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


@torch.no_grad()
def ddpm_sample(
    model: UNet,
    scheduler: NoiseScheduler,
    shape: Tuple[int, ...],
    device: str = "cpu",
) -> torch.Tensor:
    """Generate images via iterative denoising (reverse diffusion).

    Starts from pure Gaussian noise x_T and iteratively denoises
    from timestep T-1 down to 0.

    Args:
        model: Trained UNet for noise prediction.
        scheduler: NoiseScheduler with precomputed coefficients.
        shape: Shape of images to generate (B, C, H, W).
        device: Device to run on.

    Returns:
        Generated images of the specified shape.
    """
    model.eval()

    # Start from pure noise: x_T ~ N(0, I)
    x = torch.randn(shape, device=device)

    # Iterative denoising from T-1 to 0
    for t in reversed(range(scheduler.num_timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise at current timestep
        predicted_noise = model(x, t_batch)

        # Reverse diffusion step: x_{t-1} = ...
        x = scheduler.step(predicted_noise, t, x)

    return x
