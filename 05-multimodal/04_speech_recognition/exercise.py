"""Exercise: Implement a Whisper-style Speech Recognition model.

Complete the TODOs below to build an encoder-decoder speech recognition model.
Run `pytest tests.py` to verify your implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


@dataclass
class WhisperConfig:
    """Configuration for the Whisper-style speech recognition model.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft: Number of DFT frequency bins (feature_dim = n_fft // 2 + 1).
        frame_length: Number of audio samples per frame.
        hop_length: Number of audio samples between adjacent frames.
        d_model: Model dimensionality (shared between encoder and decoder).
        num_encoder_layers: Number of transformer encoder blocks.
        num_decoder_layers: Number of transformer decoder blocks.
        num_heads: Number of attention heads.
        dim_feedforward: Hidden dimension for feed-forward networks.
        max_source_positions: Maximum number of audio frames (encoder seq len).
        max_target_positions: Maximum number of decoder tokens.
        vocab_size: Output vocabulary size.
        dropout: Dropout probability.
    """

    sample_rate: int = 16000
    n_fft: int = 128
    frame_length: int = 256
    hop_length: int = 128
    d_model: int = 64
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 256
    max_source_positions: int = 1024
    max_target_positions: int = 512
    vocab_size: int = 256
    dropout: float = 0.0


class AudioFeatureExtractorExercise(nn.Module):
    """Convert raw audio waveforms to spectrogram-like features.

    TODO: Implement the __init__ and forward methods.

    This is a simplified audio preprocessing pipeline:
    1. Frame the audio into overlapping windows.
    2. Apply a Hann window and compute the DFT magnitude spectrum.
    3. Take log(1 + magnitude) for numerical stability.
    4. Project frequency bins to the model dimension.

    Args:
        config: A WhisperConfig with audio processing parameters.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft
        feature_dim = config.n_fft // 2 + 1

        # TODO 1: Create a linear projection from feature_dim to d_model
        # This maps frequency bins to the model dimension
        self.projection = None  # YOUR CODE HERE

        # TODO 2: Create LayerNorm for the output dimension (d_model)
        self.norm = None  # YOUR CODE HERE

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from raw audio.

        Args:
            audio: Raw audio waveform of shape (batch_size, num_samples).

        Returns:
            Feature matrix of shape (batch_size, num_frames, d_model).
        """
        B, num_samples = audio.shape

        # Step 1: Frame the audio into overlapping windows
        num_frames = max(1, (num_samples - self.frame_length) // self.hop_length + 1)
        indices = torch.arange(0, self.frame_length, device=audio.device).unsqueeze(0) + \
                  torch.arange(0, num_frames, device=audio.device).unsqueeze(1) * self.hop_length
        indices = indices.clamp(max=num_samples - 1)

        frames = audio.unsqueeze(1).expand(-1, num_frames, -1)
        frames = torch.gather(frames, 2, indices.unsqueeze(0).expand(B, -1, -1))

        # Step 2: Apply Hann window
        window = torch.hann_window(self.frame_length, device=audio.device)
        frames = frames * window.unsqueeze(0).unsqueeze(0)

        # Step 3: Compute DFT magnitude spectrum
        if self.frame_length < self.n_fft:
            frames = F.pad(frames, (0, self.n_fft - self.frame_length))

        # TODO 3: Compute the real FFT of the frames
        # Hint: torch.fft.rfft(frames, n=self.n_fft)
        spectrum = None  # YOUR CODE HERE

        # TODO 4: Take the magnitude (absolute value) of the spectrum
        # Hint: torch.abs(spectrum)
        magnitude = None  # YOUR CODE HERE

        # Step 4: Log scaling
        features = torch.log1p(magnitude)

        # Step 5: Project to model dimension
        # TODO 5: Apply the linear projection, layer norm, and dropout
        # features = self.projection(features)
        # features = self.norm(features)
        # features = self.dropout(features)

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
        # TODO 6: Create LayerNorm for attention sub-layer
        self.norm1 = None  # YOUR CODE HERE

        # TODO 7: Create MultiheadAttention (batch_first=True)
        self.attn = None  # YOUR CODE HERE

        # TODO 8: Create LayerNorm for FFN sub-layer
        self.norm2 = None  # YOUR CODE HERE

        # TODO 9: Create the FFN (Linear -> GELU -> Dropout -> Linear -> Dropout)
        self.ffn = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder block.

        Args:
            x: Input of shape (batch_size, seq_len, d_model).

        Returns:
            Output of shape (batch_size, seq_len, d_model).
        """
        # TODO 10: Pre-Norm Self-Attention with residual connection
        # normed = self.norm1(x)
        # attn_out, _ = self.attn(normed, normed, normed)
        # x = x + attn_out

        # TODO 11: Pre-Norm FFN with residual connection
        # x = x + self.ffn(self.norm2(x))

        return None  # YOUR CODE HERE


class DecoderBlockExercise(nn.Module):
    """Pre-Norm Transformer decoder block with cross-attention.

    TODO: Implement the __init__ and forward methods.

    Architecture:
        x = x + CausalSelfAttention(LayerNorm(x))
        x = x + CrossAttention(LayerNorm(x), encoder_output)
        x = x + FFN(LayerNorm(x))

    Cross-attention uses the decoder as Q and encoder output as K, V.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        # TODO 12: Create LayerNorm + MultiheadAttention for causal self-attention
        self.norm1 = None  # YOUR CODE HERE
        self.self_attn = None  # YOUR CODE HERE

        # TODO 13: Create LayerNorm + MultiheadAttention for cross-attention
        # This attends to the encoder output
        self.norm2 = None  # YOUR CODE HERE
        self.cross_attn = None  # YOUR CODE HERE

        # TODO 14: Create LayerNorm + FFN for the feed-forward network
        self.norm3 = None  # YOUR CODE HERE
        self.ffn = None  # YOUR CODE HERE

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder block.

        Args:
            x: Decoder input of shape (batch_size, out_len, d_model).
            encoder_output: Encoder output of shape
                (batch_size, num_frames, d_model). If None, cross-attention
                is skipped.
            causal_mask: Boolean causal mask of shape (out_len, out_len).
                True values are masked (not attended to).

        Returns:
            Output of shape (batch_size, out_len, d_model).
        """
        # TODO 15: Pre-Norm Causal Self-Attention with residual
        # normed = self.norm1(x)
        # attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask)
        # x = x + attn_out

        # TODO 16: Pre-Norm Cross-Attention with residual
        # if encoder_output is not None:
        #     normed = self.norm2(x)
        #     cross_out, _ = self.cross_attn(normed, encoder_output, encoder_output)
        #     x = x + cross_out

        # TODO 17: Pre-Norm FFN with residual
        # x = x + self.ffn(self.norm3(x))

        return None  # YOUR CODE HERE


class WhisperEncoderExercise(nn.Module):
    """Transformer encoder for audio features.

    TODO: Implement the __init__ and forward methods.

    Architecture:
        audio_features (B, num_frames, d_model)
            -> Add positional encoding
            -> N x EncoderBlock (self-attention + FFN)
            -> LayerNorm
            -> encoded output (B, num_frames, d_model)

    Args:
        config: A WhisperConfig with encoder parameters.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model

        # TODO 18: Create learnable positional encoding
        # Hint: nn.Parameter(torch.zeros(config.max_source_positions, config.d_model))
        self.positional_encoding = None  # YOUR CODE HERE

        self.dropout = nn.Dropout(config.dropout)

        # TODO 19: Create a ModuleList of EncoderBlockExercise layers
        # Hint: Use a list comprehension with config.num_encoder_layers iterations
        self.layers = None  # YOUR CODE HERE

        # TODO 20: Create the final LayerNorm
        self.norm = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio features.

        Args:
            x: Audio features of shape (batch_size, num_frames, d_model).

        Returns:
            Encoded representations of shape (batch_size, num_frames, d_model).
        """
        seq_len = x.shape[1]

        # TODO 21: Add positional encoding and apply dropout
        # x = x + self.positional_encoding[:seq_len]
        # x = self.dropout(x)

        # TODO 22: Process through encoder blocks
        # for layer in self.layers:
        #     x = layer(x)

        # TODO 23: Apply final normalization
        # x = self.norm(x)

        return None  # YOUR CODE HERE


class WhisperDecoderExercise(nn.Module):
    """Autoregressive text decoder with cross-attention.

    TODO: Implement the __init__ and forward methods.

    Args:
        config: A WhisperConfig with decoder parameters.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads

        # TODO 24: Create token embedding (vocab_size -> d_model)
        self.token_embedding = None  # YOUR CODE HERE

        # TODO 25: Create learnable positional encoding for decoder
        # Hint: nn.Parameter(torch.zeros(config.max_target_positions, config.d_model))
        self.positional_encoding = None  # YOUR CODE HERE

        self.dropout = nn.Dropout(config.dropout)

        # TODO 26: Create a ModuleList of DecoderBlockExercise layers
        self.layers = None  # YOUR CODE HERE

        # TODO 27: Create final LayerNorm
        self.norm = None  # YOUR CODE HERE

        # TODO 28: Create output projection (d_model -> vocab_size)
        self.output_projection = None  # YOUR CODE HERE

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token IDs into logits.

        Args:
            decoder_input_ids: Token IDs of shape (batch_size, out_len).
            encoder_output: Encoder output of shape
                (batch_size, num_frames, d_model).

        Returns:
            Logits of shape (batch_size, out_len, vocab_size).
        """
        B, out_len = decoder_input_ids.shape

        # TODO 29: Compute token + positional embeddings and apply dropout
        # x = self.token_embedding(decoder_input_ids)
        # x = x + self.positional_encoding[:out_len]
        # x = self.dropout(x)

        # Build causal attention mask
        causal_mask = torch.triu(
            torch.ones(out_len, out_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        # TODO 30: Process through decoder blocks
        # for layer in self.layers:
        #     x = layer(x, encoder_output, causal_mask)

        # TODO 31: Apply final normalization and output projection
        # x = self.norm(x)
        # logits = self.output_projection(x)

        return None  # YOUR CODE HERE


class WhisperModelExercise(nn.Module):
    """End-to-end Whisper-style speech recognition model.

    TODO: Implement the __init__ and forward methods.

    Combines AudioFeatureExtractor, WhisperEncoder, and WhisperDecoder.

    Args:
        config: A WhisperConfig with all model parameters.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        # TODO 32: Create the AudioFeatureExtractorExercise
        self.feature_extractor = None  # YOUR CODE HERE

        # TODO 33: Create the WhisperEncoderExercise
        self.encoder = None  # YOUR CODE HERE

        # TODO 34: Create the WhisperDecoderExercise
        self.decoder = None  # YOUR CODE HERE

    def forward(
        self,
        audio: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: encode audio and decode text.

        Args:
            audio: Raw audio waveform of shape (batch_size, num_samples).
            decoder_input_ids: Decoder token IDs of shape
                (batch_size, out_len).

        Returns:
            Logits of shape (batch_size, out_len, vocab_size).
        """
        # TODO 35: Extract audio features from raw audio
        # audio_features = self.feature_extractor(audio)

        # TODO 36: Encode audio features
        # encoder_output = self.encoder(audio_features)

        # TODO 37: Decode to token logits
        # logits = self.decoder(decoder_input_ids, encoder_output)

        return None  # YOUR CODE HERE
