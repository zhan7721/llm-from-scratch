"""Reference solution for the Text-to-Speech generation exercise."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TTSConfig:
    """Configuration for the Text-to-Speech model."""

    vocab_size: int = 256
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    max_text_positions: int = 512
    n_mels: int = 80
    vocoder_channels: int = 64
    vocoder_num_layers: int = 4
    hop_length: int = 256
    dropout: float = 0.0


class EncoderBlockSolution(nn.Module):
    """Pre-Norm Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder block."""
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))
        return x


class TextEncoderSolution(nn.Module):
    """Encode text tokens into hidden representations."""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.d_model = config.d_model

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(config.max_text_positions, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            EncoderBlockSolution(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text token IDs."""
        seq_len = text_ids.shape[1]

        x = self.token_embedding(text_ids)
        x = x + self.positional_encoding[:seq_len]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class DurationPredictorSolution(nn.Module):
    """Predict the number of mel frames per text token."""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.proj = nn.Linear(config.d_model, 1)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Predict durations from encoder hidden states."""
        durations = self.proj(encoder_output).squeeze(-1)
        durations = F.relu(durations)
        return durations


class DilatedConvLayerSolution(nn.Module):
    """Dilated causal convolution layer with gated activation."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels * 2,
            kernel_size=3, padding=dilation, dilation=dilation,
        )
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the dilated conv layer."""
        residual = x

        out = self.conv(x)
        out = out[:, :, :num_samples]

        tanh_out, sigmoid_out = out.chunk(2, dim=1)
        gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)

        residual_out = self.residual_conv(gated) + residual
        skip_out = self.skip_conv(gated)

        return residual_out, skip_out


class VocoderSolution(nn.Module):
    """Generate waveform from mel spectrogram."""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.hop_length = config.hop_length

        self.input_proj = nn.Linear(config.n_mels, config.vocoder_channels)

        self.layers = nn.ModuleList([
            DilatedConvLayerSolution(config.vocoder_channels, dilation=2 ** i)
            for i in range(config.vocoder_num_layers)
        ])

        self.output_proj = nn.Linear(config.vocoder_channels, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram."""
        B, n_mels, num_frames = mel.shape

        x = mel.transpose(1, 2)
        x = self.input_proj(x)
        x = x.repeat_interleave(self.hop_length, dim=1)
        x = x.transpose(1, 2)
        num_samples = x.shape[2]

        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x, num_samples)
            skip_connections.append(skip)

        # Sum skip connections and add final residual output
        x = sum(skip_connections) + x
        x = x.transpose(1, 2)
        waveform = self.output_proj(x).squeeze(-1)

        return waveform


class SimpleTTSSolution(nn.Module):
    """End-to-end text-to-speech pipeline."""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        self.encoder = TextEncoderSolution(config)
        self.duration_predictor = DurationPredictorSolution(config)
        self.mel_proj = nn.Linear(config.d_model, config.n_mels)
        self.vocoder = VocoderSolution(config)

    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: predict mel spectrogram from text."""
        encoder_out = self.encoder(text_ids)
        durations = self.duration_predictor(encoder_out)

        if target_mel is not None:
            num_frames = target_mel.shape[2]
        else:
            num_frames = int(durations.sum(dim=1).mean().item())
            num_frames = max(num_frames, 1)

        encoder_out_t = encoder_out.transpose(1, 2)
        expanded = F.interpolate(
            encoder_out_t, size=num_frames, mode="linear", align_corners=False,
        )
        expanded = expanded.transpose(1, 2)

        predicted_mel = self.mel_proj(expanded)
        predicted_mel = predicted_mel.transpose(1, 2)

        return predicted_mel

    @torch.no_grad()
    def synthesize(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Generate a waveform from text (inference only).

        Puts the model in evaluation mode, predicts a mel spectrogram,
        and passes it through the vocoder to produce a waveform.

        Args:
            text_ids: Text token IDs of shape (batch_size, seq_len).

        Returns:
            Waveform of shape (batch_size, num_samples).
        """
        was_training = self.training
        self.training = False

        predicted_mel = self.forward(text_ids)
        waveform = self.vocoder(predicted_mel)

        self.training = was_training
        return waveform


class TTSLossSolution(nn.Module):
    """MSE loss on mel spectrogram prediction.

    Computes the mean squared error between the predicted and target mel
    spectrograms.

    Args:
        None.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            predicted_mel: Predicted mel of shape (batch_size, n_mels, num_frames).
            target_mel: Target mel of shape (batch_size, n_mels, num_frames).

        Returns:
            Scalar loss tensor.
        """
        return F.mse_loss(predicted_mel, target_mel)
