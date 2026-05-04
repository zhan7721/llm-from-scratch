"""Whisper-style Speech Recognition implementation.

This module implements a simplified version of the Whisper speech recognition
model (Radford et al., 2022). It contains:
- AudioFeatureExtractor: Converts raw audio waveforms to spectrogram features
  using windowed DFT (a simplified mel spectrogram).
- WhisperEncoder: Transformer encoder that processes audio features with
  Pre-Norm blocks and positional embeddings.
- WhisperDecoder: Autoregressive text decoder with causal self-attention and
  cross-attention to encoder outputs.
- WhisperModel: End-to-end encoder-decoder that combines all components.

Architecture overview:
    raw audio (B, num_samples)
        -> AudioFeatureExtractor    (B, num_frames, d_model)
        -> WhisperEncoder           (B, num_frames, d_model)
        -> WhisperDecoder           (B, out_len, vocab_size)

Design notes:
- This module is self-contained (no imports from other chapters).
- Uses simplified audio processing: windowed DFT instead of real mel filters.
- Uses Pre-Norm transformer blocks (LayerNorm before attention/FFN).
- Cross-attention in decoder: K, V from encoder output; Q from decoder.
- Greedy decoding for the transcribe method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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


class AudioFeatureExtractor(nn.Module):
    """Convert raw audio waveforms to spectrogram-like features.

    This is a simplified version of Whisper's audio preprocessing. Instead of
    using librosa for mel spectrograms, it implements:
    1. Framing: Split audio into overlapping windows.
    2. Windowed DFT: Apply a Hann window and compute the magnitude spectrum.
    3. Log scaling: Take log(1 + magnitude) for numerical stability.
    4. Linear projection: Map frequency bins to the model dimension.

    Args:
        config: A WhisperConfig with audio processing parameters.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft
        feature_dim = config.n_fft // 2 + 1

        # Learnable projection from frequency bins to model dimension
        self.projection = nn.Linear(feature_dim, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
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
        # Create indices for each frame
        num_frames = max(1, (num_samples - self.frame_length) // self.hop_length + 1)
        indices = torch.arange(0, self.frame_length, device=audio.device).unsqueeze(0) + \
                  torch.arange(0, num_frames, device=audio.device).unsqueeze(1) * self.hop_length
        indices = indices.clamp(max=num_samples - 1)

        # (B, num_samples) -> (B, num_frames, frame_length)
        frames = audio.unsqueeze(1).expand(-1, num_frames, -1)
        frames = torch.gather(
            frames, 2,
            indices.unsqueeze(0).expand(B, -1, -1)
        )

        # Step 2: Apply Hann window to reduce spectral leakage
        window = torch.hann_window(self.frame_length, device=audio.device)
        frames = frames * window.unsqueeze(0).unsqueeze(0)

        # Step 3: Compute DFT magnitude spectrum
        # Pad frames to n_fft if needed
        if self.frame_length < self.n_fft:
            frames = F.pad(frames, (0, self.n_fft - self.frame_length))

        # Compute real FFT and take magnitude
        # (B, num_frames, n_fft) -> (B, num_frames, n_fft//2 + 1)
        spectrum = torch.fft.rfft(frames, n=self.n_fft)
        magnitude = torch.abs(spectrum)

        # Step 4: Log scaling for numerical stability
        features = torch.log1p(magnitude)

        # Step 5: Project to model dimension
        # (B, num_frames, feature_dim) -> (B, num_frames, d_model)
        features = self.projection(features)
        features = self.norm(features)
        features = self.dropout(features)

        return features


class WhisperEncoder(nn.Module):
    """Transformer encoder for audio features.

    Processes audio features through positional encoding and transformer
    encoder blocks. Uses Pre-Norm (LayerNorm before attention/FFN) following
    the Whisper architecture.

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

        # Learnable positional embeddings for audio frames
        self.positional_encoding = nn.Parameter(
            torch.zeros(config.max_source_positions, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        # Stack of transformer encoder blocks
        self.layers = nn.ModuleList([
            _EncoderBlock(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio features.

        Args:
            x: Audio features of shape (batch_size, num_frames, d_model).

        Returns:
            Encoded representations of shape (batch_size, num_frames, d_model).
        """
        seq_len = x.shape[1]

        # Add positional encoding (broadcast over batch dimension)
        x = x + self.positional_encoding[:seq_len]
        x = self.dropout(x)

        # Process through encoder blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        return x


class WhisperDecoder(nn.Module):
    """Autoregressive text decoder with cross-attention to encoder output.

    Takes token IDs and encoder output, applies causal self-attention and
    cross-attention to produce logits over the vocabulary.

    Architecture:
        token_ids (B, out_len)
            -> Token + Position embedding
            -> N x DecoderBlock (causal self-attn + cross-attn + FFN)
            -> LayerNorm
            -> Linear projection to vocab
            -> logits (B, out_len, vocab_size)

    Args:
        config: A WhisperConfig with decoder parameters.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(config.max_target_positions, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        # Stack of transformer decoder blocks with cross-attention
        self.layers = nn.ModuleList([
            _DecoderBlock(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])

        # Final layer normalization and output projection
        self.norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token IDs into logits.

        Args:
            decoder_input_ids: Token IDs of shape (batch_size, out_len).
            encoder_output: Encoder output of shape
                (batch_size, num_frames, d_model). If None, cross-attention
                is skipped (useful for testing the decoder independently).

        Returns:
            Logits of shape (batch_size, out_len, vocab_size).
        """
        B, out_len = decoder_input_ids.shape

        # Token + positional embeddings
        x = self.token_embedding(decoder_input_ids)
        x = x + self.positional_encoding[:out_len]
        x = self.dropout(x)

        # Build causal attention mask (upper triangular = masked)
        # Positions cannot attend to future positions
        causal_mask = torch.triu(
            torch.ones(out_len, out_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        # Process through decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask)

        # Final normalization and projection to vocabulary
        x = self.norm(x)
        logits = self.output_projection(x)

        return logits


class WhisperModel(nn.Module):
    """End-to-end Whisper-style speech recognition model.

    Combines AudioFeatureExtractor, WhisperEncoder, and WhisperDecoder into
    a complete speech-to-text pipeline.

    Architecture:
        raw audio (B, num_samples)
            -> AudioFeatureExtractor    (B, num_frames, d_model)
            -> WhisperEncoder           (B, num_frames, d_model)
            -> WhisperDecoder           (B, out_len, vocab_size)

    Args:
        config: A WhisperConfig with all model parameters.
    """

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        # Audio preprocessing (simplified mel spectrogram)
        self.feature_extractor = AudioFeatureExtractor(config)

        # Encoder: processes audio features
        self.encoder = WhisperEncoder(config)

        # Decoder: generates text tokens
        self.decoder = WhisperDecoder(config)

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
        # Extract audio features
        audio_features = self.feature_extractor(audio)

        # Encode audio features
        encoder_output = self.encoder(audio_features)

        # Decode to token logits
        logits = self.decoder(decoder_input_ids, encoder_output)

        return logits

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, max_len: int = 448) -> torch.Tensor:
        """Transcribe audio to token IDs using greedy decoding.

        Generates tokens one at a time, always selecting the token with the
        highest probability. This is the simplest decoding strategy.

        Note: Puts the model in evaluation mode before generation.

        Args:
            audio: Raw audio waveform of shape (batch_size, num_samples).
            max_len: Maximum number of tokens to generate.

        Returns:
            Generated token IDs of shape (batch_size, generated_len).
        """
        was_training = self.training
        self.eval()
        B = audio.shape[0]
        device = audio.device

        # Encode audio once
        audio_features = self.feature_extractor(audio)
        encoder_output = self.encoder(audio_features)

        # Start with token ID 0 as the initial decoder input
        generated = torch.zeros(B, 1, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            # Get logits for the current sequence
            logits = self.decoder(generated, encoder_output)

            # Greedy: take the token with highest logit at the last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to the generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        # Restore original training mode
        self.train(was_training)

        return generated


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


class _DecoderBlock(nn.Module):
    """Pre-Norm Transformer decoder block with cross-attention.

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
        # Causal self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Cross-attention to encoder output
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        # Feed-forward network
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

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
        # Pre-Norm Causal Self-Attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            normed, normed, normed, attn_mask=causal_mask,
        )
        x = x + attn_out

        # Pre-Norm Cross-Attention with residual
        if encoder_output is not None:
            normed = self.norm2(x)
            # Q from decoder, K and V from encoder
            cross_out, _ = self.cross_attn(normed, encoder_output, encoder_output)
            x = x + cross_out

        # Pre-Norm FFN with residual
        x = x + self.ffn(self.norm3(x))

        return x
