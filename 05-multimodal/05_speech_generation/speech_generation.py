"""Text-to-Speech (TTS) generation implementation.

This module implements a simplified TTS model inspired by architectures like
Tacotron and WaveNet. It contains:
- TextEncoder: Encodes text tokens into hidden representations using a
  Transformer encoder with Pre-Norm blocks and learned positional encoding.
- DurationPredictor: Predicts the number of mel frames each text token
  should produce (a simple linear layer).
- Vocoder: Converts a mel spectrogram into a raw waveform using a
  simplified WaveNet-style architecture with dilated convolutions and
  gated activations.
- SimpleTTS: End-to-end pipeline that combines TextEncoder, DurationPredictor,
  and Vocoder.
- TTSLoss: MSE loss on mel spectrogram prediction.

Architecture overview:
    text_ids (B, seq_len)
        -> TextEncoder              (B, seq_len, d_model)
        -> DurationPredictor        (B, seq_len)
        -> Expand / Interpolate     (B, num_frames, d_model)
        -> Linear projection        (B, n_mels, num_frames)  [predicted mel]
        -> Vocoder                  (B, num_samples)          [waveform]

Design notes:
- This module is self-contained (no imports from other chapters).
- Uses Pre-Norm transformer blocks (LayerNorm before attention/FFN).
- The vocoder uses dilated causal convolutions with gated activations
  (tanh * sigmoid) for a large receptive field, inspired by WaveNet.
- During training, encoder outputs are interpolated to match the target
  mel spectrogram length. During inference, predicted durations determine
  the output length.
- No external dependencies beyond PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TTSConfig:
    """Configuration for the Text-to-Speech model.

    Args:
        vocab_size: Size of the text vocabulary.
        d_model: Model dimensionality (shared across components).
        num_heads: Number of attention heads in the TextEncoder.
        num_layers: Number of transformer encoder blocks in TextEncoder.
        dim_feedforward: Hidden dimension for feed-forward networks.
        max_text_positions: Maximum text sequence length (for positional encoding).
        n_mels: Number of mel spectrogram frequency bins.
        vocoder_channels: Number of channels in the vocoder's hidden layers.
        vocoder_num_layers: Number of dilated conv layers in the vocoder.
        hop_length: Audio samples per mel frame (used for vocoder upsampling).
        dropout: Dropout probability.
    """

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


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Encode text tokens into hidden representations.

    Architecture:
        text_ids (B, seq_len)
            -> Token embedding + positional encoding
            -> N x Pre-Norm EncoderBlock (self-attention + FFN)
            -> LayerNorm
            -> hidden states (B, seq_len, d_model)

    Args:
        config: A TTSConfig with encoder parameters.
    """

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.d_model = config.d_model

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(config.max_text_positions, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        # Stack of Pre-Norm transformer encoder blocks
        self.layers = nn.ModuleList([
            _EncoderBlock(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text token IDs.

        Args:
            text_ids: Token IDs of shape (batch_size, seq_len).

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
        """
        seq_len = text_ids.shape[1]

        # Token + positional embeddings
        x = self.token_embedding(text_ids)
        x = x + self.positional_encoding[:seq_len]
        x = self.dropout(x)

        # Process through encoder blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        return x


# ---------------------------------------------------------------------------
# DurationPredictor
# ---------------------------------------------------------------------------


class DurationPredictor(nn.Module):
    """Predict the number of mel frames per text token.

    A simple linear projection from d_model to 1, followed by ReLU to
    ensure non-negative durations.

    Args:
        config: A TTSConfig with model dimension.
    """

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.proj = nn.Linear(config.d_model, 1)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Predict durations from encoder hidden states.

        Args:
            encoder_output: Hidden states of shape (batch_size, seq_len, d_model).

        Returns:
            Durations of shape (batch_size, seq_len). Each value is >= 0.
        """
        # (B, seq_len, d_model) -> (B, seq_len, 1) -> (B, seq_len)
        durations = self.proj(encoder_output).squeeze(-1)
        durations = F.relu(durations)
        return durations


# ---------------------------------------------------------------------------
# Vocoder
# ---------------------------------------------------------------------------


class Vocoder(nn.Module):
    """Generate waveform from mel spectrogram.

    A simplified WaveNet-style vocoder that upsamples a mel spectrogram
    to waveform samples using dilated causal convolutions with gated
    activations (tanh * sigmoid).

    Architecture:
        mel (B, n_mels, num_frames)
            -> Upsample (repeat_interleave by hop_length)
            -> Input projection (n_mels -> vocoder_channels)
            -> N x DilatedConvLayer (gated conv + residual + skip)
            -> Sum skip connections
            -> Output projection (vocoder_channels -> 1)
            -> waveform (B, num_samples)

    Args:
        config: A TTSConfig with vocoder parameters.
    """

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.hop_length = config.hop_length

        # Input projection: n_mels -> vocoder_channels
        self.input_proj = nn.Linear(config.n_mels, config.vocoder_channels)

        # Stack of dilated conv layers with increasing dilation
        self.layers = nn.ModuleList([
            _DilatedConvLayer(config.vocoder_channels, dilation=2 ** i)
            for i in range(config.vocoder_num_layers)
        ])

        # Output projection: vocoder_channels -> 1 (waveform sample)
        self.output_proj = nn.Linear(config.vocoder_channels, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram of shape (batch_size, n_mels, num_frames).

        Returns:
            Waveform of shape (batch_size, num_samples).
        """
        B, n_mels, num_frames = mel.shape

        # Upsample: (B, n_mels, num_frames) -> (B, num_frames, n_mels)
        x = mel.transpose(1, 2)

        # Project: (B, num_frames, n_mels) -> (B, num_frames, channels)
        x = self.input_proj(x)

        # Repeat to upsample: (B, num_frames, channels) -> (B, num_samples, channels)
        x = x.repeat_interleave(self.hop_length, dim=1)

        # Transpose for Conv1d: (B, channels, num_samples)
        x = x.transpose(1, 2)
        num_samples = x.shape[2]

        # Process through dilated conv layers, accumulating skip connections
        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x, num_samples)
            skip_connections.append(skip)

        # Sum all skip connections and add final residual output
        x = sum(skip_connections) + x

        # Project to waveform: (B, channels, num_samples) -> (B, num_samples, channels)
        x = x.transpose(1, 2)
        # (B, num_samples, channels) -> (B, num_samples, 1) -> (B, num_samples)
        waveform = self.output_proj(x).squeeze(-1)

        return waveform


# ---------------------------------------------------------------------------
# SimpleTTS
# ---------------------------------------------------------------------------


class SimpleTTS(nn.Module):
    """End-to-end text-to-speech pipeline.

    Combines TextEncoder, DurationPredictor, and Vocoder to convert text
    token IDs into either a predicted mel spectrogram (training) or a
    raw waveform (inference).

    Architecture:
        text_ids (B, seq_len)
            -> TextEncoder          (B, seq_len, d_model)
            -> DurationPredictor    (B, seq_len)
            -> Expand / Interpolate (B, num_frames, d_model)
            -> Linear projection    (B, n_mels, num_frames)
            -> Vocoder              (B, num_samples)  [inference only]

    Args:
        config: A TTSConfig with all model parameters.
    """

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        # Text encoder: text tokens -> hidden representations
        self.encoder = TextEncoder(config)

        # Duration predictor: hidden states -> frame counts per token
        self.duration_predictor = DurationPredictor(config)

        # Mel projection: d_model -> n_mels (predict mel spectrogram)
        self.mel_proj = nn.Linear(config.d_model, config.n_mels)

        # Vocoder: mel spectrogram -> waveform
        self.vocoder = Vocoder(config)

    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: predict mel spectrogram from text.

        During training, pass target_mel to determine the output length.
        During inference (target_mel=None), use predicted durations.

        Args:
            text_ids: Text token IDs of shape (batch_size, seq_len).
            target_mel: Target mel spectrogram of shape
                (batch_size, n_mels, num_frames). If None, uses predicted
                durations to determine output length.

        Returns:
            Predicted mel spectrogram of shape (batch_size, n_mels, num_frames).
        """
        # Encode text
        encoder_out = self.encoder(text_ids)  # (B, seq_len, d_model)

        # Predict durations
        durations = self.duration_predictor(encoder_out)  # (B, seq_len)

        # Determine output length
        if target_mel is not None:
            num_frames = target_mel.shape[2]
        else:
            # Use predicted durations to determine total frame count
            num_frames = int(durations.sum(dim=1).mean().item())
            num_frames = max(num_frames, 1)

        # Expand encoder output to match mel frame count
        # (B, seq_len, d_model) -> (B, d_model, seq_len) for interpolation
        encoder_out_t = encoder_out.transpose(1, 2)
        # Interpolate: (B, d_model, seq_len) -> (B, d_model, num_frames)
        expanded = F.interpolate(
            encoder_out_t, size=num_frames, mode="linear", align_corners=False,
        )
        # (B, d_model, num_frames) -> (B, num_frames, d_model)
        expanded = expanded.transpose(1, 2)

        # Project to mel spectrogram
        # (B, num_frames, d_model) -> (B, num_frames, n_mels)
        predicted_mel = self.mel_proj(expanded)

        # Transpose to (B, n_mels, num_frames) for vocoder compatibility
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
        self.eval()

        predicted_mel = self.forward(text_ids)
        waveform = self.vocoder(predicted_mel)

        self.train(was_training)
        return waveform


# ---------------------------------------------------------------------------
# TTSLoss
# ---------------------------------------------------------------------------


class TTSLoss(nn.Module):
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


# ---------------------------------------------------------------------------
# Internal building blocks (not exported in __all__)
# ---------------------------------------------------------------------------


class _EncoderBlock(nn.Module):
    """Pre-Norm Transformer encoder block.

    Architecture:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

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
        """Run the encoder block.

        Args:
            x: Input of shape (batch_size, seq_len, d_model).

        Returns:
            Output of shape (batch_size, seq_len, d_model).
        """
        # Pre-Norm Self-Attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-Norm FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class _DilatedConvLayer(nn.Module):
    """Dilated causal convolution layer with gated activation.

    Implements a single layer of a WaveNet-style vocoder:
        1. Dilated causal Conv1d (produces channels*2 outputs)
        2. Split into tanh and sigmoid branches (gated activation)
        3. Residual connection (input + gated output projected back)
        4. Skip connection (gated output projected to channels)

    Args:
        channels: Number of input/output channels.
        dilation: Dilation factor for the convolution.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # Dilated causal convolution: channels -> channels*2 for gating
        self.conv = nn.Conv1d(
            channels, channels * 2,
            kernel_size=3, padding=dilation, dilation=dilation,
        )
        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the dilated conv layer.

        Args:
            x: Input of shape (batch_size, channels, num_samples).
            num_samples: Expected output length (for trimming causal padding).

        Returns:
            Tuple of (residual_output, skip_output), each
            (batch_size, channels, num_samples).
        """
        residual = x

        # Dilated causal convolution
        out = self.conv(x)
        # Trim to remove causal padding (keep only the first num_samples)
        out = out[:, :, :num_samples]

        # Gated activation: tanh(branch1) * sigmoid(branch2)
        tanh_out, sigmoid_out = out.chunk(2, dim=1)
        gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)

        # Residual connection
        residual_out = self.residual_conv(gated) + residual

        # Skip connection
        skip_out = self.skip_conv(gated)

        return residual_out, skip_out
