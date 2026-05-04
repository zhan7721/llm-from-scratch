"""Reference solution for the Whisper-style Speech Recognition exercise."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class WhisperConfig:
    """Configuration for the Whisper-style speech recognition model."""

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


class AudioFeatureExtractorSolution(nn.Module):
    """Convert raw audio waveforms to spectrogram-like features."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft
        feature_dim = config.n_fft // 2 + 1

        # Linear projection from frequency bins to model dimension
        self.projection = nn.Linear(feature_dim, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from raw audio."""
        B, num_samples = audio.shape

        # Frame the audio into overlapping windows
        num_frames = max(1, (num_samples - self.frame_length) // self.hop_length + 1)
        indices = torch.arange(0, self.frame_length, device=audio.device).unsqueeze(0) + \
                  torch.arange(0, num_frames, device=audio.device).unsqueeze(1) * self.hop_length
        indices = indices.clamp(max=num_samples - 1)

        frames = audio.unsqueeze(1).expand(-1, num_frames, -1)
        frames = torch.gather(frames, 2, indices.unsqueeze(0).expand(B, -1, -1))

        # Apply Hann window
        window = torch.hann_window(self.frame_length, device=audio.device)
        frames = frames * window.unsqueeze(0).unsqueeze(0)

        # Compute DFT magnitude spectrum
        if self.frame_length < self.n_fft:
            frames = F.pad(frames, (0, self.n_fft - self.frame_length))

        spectrum = torch.fft.rfft(frames, n=self.n_fft)
        magnitude = torch.abs(spectrum)

        # Log scaling
        features = torch.log1p(magnitude)

        # Project to model dimension
        features = self.projection(features)
        features = self.norm(features)
        features = self.dropout(features)

        return features


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


class DecoderBlockSolution(nn.Module):
    """Pre-Norm Transformer decoder block with cross-attention."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

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
        """Run the decoder block."""
        # Causal self-attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + attn_out

        # Cross-attention to encoder output
        if encoder_output is not None:
            normed = self.norm2(x)
            cross_out, _ = self.cross_attn(normed, encoder_output, encoder_output)
            x = x + cross_out

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class WhisperEncoderSolution(nn.Module):
    """Transformer encoder for audio features."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model

        self.positional_encoding = nn.Parameter(
            torch.zeros(config.max_source_positions, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            EncoderBlockSolution(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio features."""
        seq_len = x.shape[1]

        x = x + self.positional_encoding[:seq_len]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class WhisperDecoderSolution(nn.Module):
    """Autoregressive text decoder with cross-attention."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(config.max_target_positions, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderBlockSolution(config.d_model, config.num_heads, config.dim_feedforward, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token IDs into logits."""
        B, out_len = decoder_input_ids.shape

        x = self.token_embedding(decoder_input_ids)
        x = x + self.positional_encoding[:out_len]
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(out_len, out_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask)

        x = self.norm(x)
        logits = self.output_projection(x)
        return logits


class WhisperModelSolution(nn.Module):
    """End-to-end Whisper-style speech recognition model."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        self.feature_extractor = AudioFeatureExtractorSolution(config)
        self.encoder = WhisperEncoderSolution(config)
        self.decoder = WhisperDecoderSolution(config)

    def forward(
        self,
        audio: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: encode audio and decode text."""
        audio_features = self.feature_extractor(audio)
        encoder_output = self.encoder(audio_features)
        logits = self.decoder(decoder_input_ids, encoder_output)
        return logits
