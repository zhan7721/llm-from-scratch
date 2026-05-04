"""Exercise: Implement a Text-to-Speech (TTS) generation model.

Complete the TODOs below to build a TTS model that converts text tokens
into mel spectrograms and waveforms.
Run `pytest tests.py` to verify your implementation.
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


class TextEncoderExercise(nn.Module):
    """Encode text tokens into hidden representations.

    TODO: Implement the __init__ and forward methods.

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

        # TODO 1: Create token embedding (vocab_size -> d_model)
        self.token_embedding = None  # YOUR CODE HERE

        # TODO 2: Create learnable positional encoding
        # Hint: nn.Parameter(torch.zeros(config.max_text_positions, config.d_model))
        self.positional_encoding = None  # YOUR CODE HERE

        self.dropout = nn.Dropout(config.dropout)

        # TODO 3: Create a ModuleList of EncoderBlockExercise layers
        # Hint: Use a list comprehension with config.num_layers iterations
        self.layers = None  # YOUR CODE HERE

        # TODO 4: Create the final LayerNorm
        self.norm = None  # YOUR CODE HERE

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text token IDs.

        Args:
            text_ids: Token IDs of shape (batch_size, seq_len).

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
        """
        seq_len = text_ids.shape[1]

        # TODO 5: Compute token + positional embeddings and apply dropout
        # x = self.token_embedding(text_ids)
        # x = x + self.positional_encoding[:seq_len]
        # x = self.dropout(x)

        # TODO 6: Process through encoder blocks
        # for layer in self.layers:
        #     x = layer(x)

        # TODO 7: Apply final normalization
        # x = self.norm(x)

        return None  # YOUR CODE HERE


class EncoderBlockExercise(nn.Module):
    """Pre-Norm Transformer encoder block.

    TODO: Implement the __init__ and forward methods.

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
        # TODO 8: Create LayerNorm for attention sub-layer
        self.norm1 = None  # YOUR CODE HERE

        # TODO 9: Create MultiheadAttention (batch_first=True)
        self.attn = None  # YOUR CODE HERE

        # TODO 10: Create LayerNorm for FFN sub-layer
        self.norm2 = None  # YOUR CODE HERE

        # TODO 11: Create the FFN (Linear -> GELU -> Dropout -> Linear -> Dropout)
        self.ffn = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder block.

        Args:
            x: Input of shape (batch_size, seq_len, d_model).

        Returns:
            Output of shape (batch_size, seq_len, d_model).
        """
        # TODO 12: Pre-Norm Self-Attention with residual connection
        # normed = self.norm1(x)
        # attn_out, _ = self.attn(normed, normed, normed)
        # x = x + attn_out

        # TODO 13: Pre-Norm FFN with residual connection
        # x = x + self.ffn(self.norm2(x))

        return None  # YOUR CODE HERE


class DurationPredictorExercise(nn.Module):
    """Predict the number of mel frames per text token.

    TODO: Implement the __init__ and forward methods.

    A simple linear projection from d_model to 1, followed by ReLU to
    ensure non-negative durations.

    Args:
        config: A TTSConfig with model dimension.
    """

    def __init__(self, config: TTSConfig):
        super().__init__()
        # TODO 14: Create a linear layer (d_model -> 1)
        self.proj = None  # YOUR CODE HERE

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Predict durations from encoder hidden states.

        Args:
            encoder_output: Hidden states of shape (batch_size, seq_len, d_model).

        Returns:
            Durations of shape (batch_size, seq_len). Each value is >= 0.
        """
        # TODO 15: Apply the linear projection, squeeze, and ReLU
        # durations = self.proj(encoder_output).squeeze(-1)
        # durations = F.relu(durations)
        # return durations

        return None  # YOUR CODE HERE


class VocoderExercise(nn.Module):
    """Generate waveform from mel spectrogram.

    TODO: Implement the __init__ and forward methods.

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

        # TODO 16: Create input projection (n_mels -> vocoder_channels)
        self.input_proj = None  # YOUR CODE HERE

        # TODO 17: Create a ModuleList of DilatedConvLayerExercise layers
        # Hint: dilation should be 2**i for layer i
        self.layers = None  # YOUR CODE HERE

        # TODO 18: Create output projection (vocoder_channels -> 1)
        self.output_proj = None  # YOUR CODE HERE

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram of shape (batch_size, n_mels, num_frames).

        Returns:
            Waveform of shape (batch_size, num_samples).
        """
        B, n_mels, num_frames = mel.shape

        # TODO 19: Upsample and project
        # x = mel.transpose(1, 2)           # (B, num_frames, n_mels)
        # x = self.input_proj(x)            # (B, num_frames, channels)
        # x = x.repeat_interleave(self.hop_length, dim=1)  # (B, num_samples, channels)
        # x = x.transpose(1, 2)            # (B, channels, num_samples)

        # num_samples = x.shape[2]

        # TODO 20: Process through dilated conv layers
        # skip_connections = []
        # for layer in self.layers:
        #     x, skip = layer(x, num_samples)
        #     skip_connections.append(skip)

        # TODO 21: Sum skip connections and project to waveform
        # x = sum(skip_connections)
        # x = x.transpose(1, 2)
        # waveform = self.output_proj(x).squeeze(-1)
        # return waveform

        return None  # YOUR CODE HERE


class DilatedConvLayerExercise(nn.Module):
    """Dilated causal convolution layer with gated activation.

    TODO: Implement the __init__ and forward methods.

    Args:
        channels: Number of input/output channels.
        dilation: Dilation factor for the convolution.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # TODO 22: Create dilated causal Conv1d (channels -> channels*2)
        # kernel_size=3, padding=dilation, dilation=dilation
        self.conv = None  # YOUR CODE HERE

        # TODO 23: Create 1x1 convolutions for residual and skip connections
        self.residual_conv = None  # YOUR CODE HERE
        self.skip_conv = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the dilated conv layer.

        Args:
            x: Input of shape (batch_size, channels, num_samples).
            num_samples: Expected output length (for trimming causal padding).

        Returns:
            Tuple of (residual_output, skip_output), each
            (batch_size, channels, num_samples).
        """
        # TODO 24: Implement dilated conv with gated activation
        # residual = x
        # out = self.conv(x)
        # out = out[:, :, :num_samples]  # Trim causal padding
        #
        # # Gated activation
        # tanh_out, sigmoid_out = out.chunk(2, dim=1)
        # gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
        #
        # # Residual and skip connections
        # residual_out = self.residual_conv(gated) + residual
        # skip_out = self.skip_conv(gated)
        #
        # return residual_out, skip_out

        return None, None  # YOUR CODE HERE


class SimpleTTSExercise(nn.Module):
    """End-to-end text-to-speech pipeline.

    TODO: Implement the __init__ and forward methods.

    Combines TextEncoder, DurationPredictor, and Vocoder.

    Args:
        config: A TTSConfig with all model parameters.
    """

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        # TODO 25: Create the TextEncoderExercise
        self.encoder = None  # YOUR CODE HERE

        # TODO 26: Create the DurationPredictorExercise
        self.duration_predictor = None  # YOUR CODE HERE

        # TODO 27: Create mel projection (d_model -> n_mels)
        self.mel_proj = None  # YOUR CODE HERE

        # TODO 28: Create the VocoderExercise
        self.vocoder = None  # YOUR CODE HERE

    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: predict mel spectrogram from text.

        Args:
            text_ids: Text token IDs of shape (batch_size, seq_len).
            target_mel: Target mel spectrogram of shape
                (batch_size, n_mels, num_frames). If None, uses predicted
                durations to determine output length.

        Returns:
            Predicted mel spectrogram of shape (batch_size, n_mels, num_frames).
        """
        # TODO 29: Encode text
        # encoder_out = self.encoder(text_ids)

        # TODO 30: Predict durations
        # durations = self.duration_predictor(encoder_out)

        # TODO 31: Determine output length
        # if target_mel is not None:
        #     num_frames = target_mel.shape[2]
        # else:
        #     num_frames = int(durations.sum(dim=1).mean().item())
        #     num_frames = max(num_frames, 1)

        # TODO 32: Expand encoder output and project to mel
        # encoder_out_t = encoder_out.transpose(1, 2)
        # expanded = F.interpolate(encoder_out_t, size=num_frames, mode="linear", align_corners=False)
        # expanded = expanded.transpose(1, 2)
        # predicted_mel = self.mel_proj(expanded)
        # predicted_mel = predicted_mel.transpose(1, 2)
        # return predicted_mel

        return None  # YOUR CODE HERE


class TTSLossExercise(nn.Module):
    """MSE loss on mel spectrogram prediction.

    TODO: Implement the forward method.

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
        # TODO 33: Compute MSE loss between predicted and target mel spectrograms
        # Hint: Use F.mse_loss
        # return F.mse_loss(predicted_mel, target_mel)

        return None  # YOUR CODE HERE
