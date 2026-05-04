"""Tests for the Whisper-style Speech Recognition implementation."""

import torch
import pytest
from speech_recognition import (
    AudioFeatureExtractor,
    WhisperEncoder,
    WhisperDecoder,
    WhisperModel,
    WhisperConfig,
)


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------
def _small_config(**overrides) -> WhisperConfig:
    """Create a small WhisperConfig suitable for unit testing."""
    defaults = dict(
        sample_rate=16000,
        n_fft=64,
        frame_length=128,
        hop_length=64,
        d_model=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dim_feedforward=64,
        max_source_positions=256,
        max_target_positions=128,
        vocab_size=64,
        dropout=0.0,
    )
    defaults.update(overrides)
    return WhisperConfig(**defaults)


def _dummy_audio(config: WhisperConfig, batch_size: int = 2, duration_frames: int = 20) -> torch.Tensor:
    """Create dummy audio tensors."""
    num_samples = duration_frames * config.hop_length + config.frame_length
    return torch.randn(batch_size, num_samples)


def _dummy_decoder_ids(config: WhisperConfig, batch_size: int = 2, seq_len: int = 10) -> torch.Tensor:
    """Create dummy decoder input token IDs."""
    return torch.randint(0, config.vocab_size, (batch_size, seq_len))


# ===========================================================================
# AudioFeatureExtractor tests
# ===========================================================================

class TestAudioFeatureExtractor:
    """Tests for the AudioFeatureExtractor module."""

    def test_output_shape(self):
        """Feature extractor output should be (B, num_frames, d_model)."""
        config = _small_config()
        extractor = AudioFeatureExtractor(config)
        audio = _dummy_audio(config, batch_size=2)
        out = extractor(audio)

        assert out.dim() == 3
        assert out.shape[0] == 2
        assert out.shape[2] == config.d_model

    def test_output_shape_batch_1(self):
        """Feature extractor should work with batch_size=1."""
        config = _small_config()
        extractor = AudioFeatureExtractor(config)
        audio = _dummy_audio(config, batch_size=1)
        out = extractor(audio)

        assert out.shape[0] == 1
        assert out.shape[2] == config.d_model

    def test_num_frames_reasonable(self):
        """Number of output frames should be roughly num_samples / hop_length."""
        config = _small_config()
        extractor = AudioFeatureExtractor(config)
        audio = _dummy_audio(config, batch_size=1, duration_frames=20)
        out = extractor(audio)

        # Should produce at least a few frames
        assert out.shape[1] >= 10

    def test_gradient_flows(self):
        """Gradients should flow through the feature extractor."""
        config = _small_config()
        extractor = AudioFeatureExtractor(config)
        audio = _dummy_audio(config, batch_size=1)
        out = extractor(audio)
        loss = out.sum()
        loss.backward()

        for name, p in extractor.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_audio_lengths(self):
        """Feature extractor should handle different audio lengths."""
        config = _small_config()
        extractor = AudioFeatureExtractor(config)
        extractor.eval()

        for duration in [10, 30, 50]:
            audio = _dummy_audio(config, batch_size=1, duration_frames=duration)
            out = extractor(audio)
            assert out.dim() == 3
            assert out.shape[0] == 1
            assert out.shape[2] == config.d_model


# ===========================================================================
# WhisperEncoder tests
# ===========================================================================

class TestWhisperEncoder:
    """Tests for the WhisperEncoder module."""

    def test_output_shape(self):
        """Encoder output should be (B, num_frames, d_model)."""
        config = _small_config()
        encoder = WhisperEncoder(config)
        features = torch.randn(2, 50, config.d_model)
        out = encoder(features)

        assert out.shape == (2, 50, config.d_model)

    def test_preserves_sequence_length(self):
        """Encoder should not change sequence length."""
        config = _small_config()
        encoder = WhisperEncoder(config)

        for seq_len in [10, 50, 100]:
            features = torch.randn(1, seq_len, config.d_model)
            out = encoder(features)
            assert out.shape[1] == seq_len

    def test_gradient_flows(self):
        """Gradients should flow through the encoder."""
        config = _small_config()
        encoder = WhisperEncoder(config)
        features = torch.randn(2, 20, config.d_model)
        out = encoder(features)
        loss = out.sum()
        loss.backward()

        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self):
        """Encoder should work with different batch sizes."""
        config = _small_config()
        encoder = WhisperEncoder(config)
        encoder.eval()

        for batch_size in [1, 2, 4]:
            features = torch.randn(batch_size, 20, config.d_model)
            out = encoder(features)
            assert out.shape == (batch_size, 20, config.d_model)

    def test_num_layers(self):
        """Encoder should have the correct number of layers."""
        config = _small_config(num_encoder_layers=4)
        encoder = WhisperEncoder(config)
        assert len(encoder.layers) == 4


# ===========================================================================
# WhisperDecoder tests
# ===========================================================================

class TestWhisperDecoder:
    """Tests for the WhisperDecoder module."""

    def test_output_shape_with_encoder(self):
        """Decoder output with encoder should be (B, out_len, vocab_size)."""
        config = _small_config()
        decoder = WhisperDecoder(config)
        decoder_ids = _dummy_decoder_ids(config, batch_size=2, seq_len=10)
        encoder_out = torch.randn(2, 50, config.d_model)

        logits = decoder(decoder_ids, encoder_out)
        assert logits.shape == (2, 10, config.vocab_size)

    def test_output_shape_without_encoder(self):
        """Decoder output without encoder should be (B, out_len, vocab_size)."""
        config = _small_config()
        decoder = WhisperDecoder(config)
        decoder_ids = _dummy_decoder_ids(config, batch_size=2, seq_len=10)

        logits = decoder(decoder_ids, encoder_output=None)
        assert logits.shape == (2, 10, config.vocab_size)

    def test_gradient_flows_with_encoder(self):
        """Gradients should flow through decoder with encoder output."""
        config = _small_config()
        decoder = WhisperDecoder(config)
        decoder_ids = _dummy_decoder_ids(config, batch_size=1, seq_len=5)
        encoder_out = torch.randn(1, 20, config.d_model)

        logits = decoder(decoder_ids, encoder_out)
        loss = logits.sum()
        loss.backward()

        for name, p in decoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_gradient_flows_without_encoder(self):
        """Gradients should flow through decoder without encoder output.

        When encoder_output is None, cross-attention parameters (norm2,
        cross_attn) are skipped, so they won't have gradients. We check that
        all other parameters receive gradients.
        """
        config = _small_config()
        decoder = WhisperDecoder(config)
        decoder_ids = _dummy_decoder_ids(config, batch_size=1, seq_len=5)

        logits = decoder(decoder_ids, encoder_output=None)
        loss = logits.sum()
        loss.backward()

        # Cross-attention params are not used when encoder_output is None
        skip_patterns = {"norm2", "cross_attn"}
        for name, p in decoder.named_parameters():
            if any(pat in name for pat in skip_patterns):
                continue
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self):
        """Decoder should work with different batch sizes."""
        config = _small_config()
        decoder = WhisperDecoder(config)
        decoder.eval()

        for batch_size in [1, 2, 4]:
            decoder_ids = _dummy_decoder_ids(config, batch_size=batch_size, seq_len=5)
            encoder_out = torch.randn(batch_size, 20, config.d_model)
            logits = decoder(decoder_ids, encoder_out)
            assert logits.shape == (batch_size, 5, config.vocab_size)

    def test_different_seq_lengths(self):
        """Decoder should handle different sequence lengths."""
        config = _small_config()
        decoder = WhisperDecoder(config)
        decoder.eval()

        for seq_len in [1, 5, 20]:
            decoder_ids = _dummy_decoder_ids(config, batch_size=1, seq_len=seq_len)
            encoder_out = torch.randn(1, 30, config.d_model)
            logits = decoder(decoder_ids, encoder_out)
            assert logits.shape == (1, seq_len, config.vocab_size)


# ===========================================================================
# WhisperModel tests
# ===========================================================================

class TestWhisperModel:
    """Tests for the full WhisperModel."""

    def test_forward_shape(self):
        """Forward pass should produce (B, out_len, vocab_size) logits."""
        config = _small_config()
        model = WhisperModel(config)
        audio = _dummy_audio(config, batch_size=2)
        decoder_ids = _dummy_decoder_ids(config, batch_size=2, seq_len=10)

        logits = model(audio, decoder_ids)
        assert logits.shape == (2, 10, config.vocab_size)

    def test_transcribe_produces_token_ids(self):
        """Transcribe should produce integer token IDs."""
        config = _small_config()
        model = WhisperModel(config)
        audio = _dummy_audio(config, batch_size=1)

        tokens = model.transcribe(audio, max_len=20)
        assert tokens.dim() == 2
        assert tokens.shape[0] == 1
        assert tokens.shape[1] == 20
        # All values should be valid vocab indices
        assert (tokens >= 0).all()
        assert (tokens < config.vocab_size).all()

    def test_transcribe_batch(self):
        """Transcribe should work with multiple inputs."""
        config = _small_config()
        model = WhisperModel(config)
        audio = _dummy_audio(config, batch_size=3)

        tokens = model.transcribe(audio, max_len=10)
        assert tokens.shape == (3, 10)

    def test_gradient_flows_full_model(self):
        """Gradients should flow through the entire model."""
        config = _small_config()
        model = WhisperModel(config)
        audio = _dummy_audio(config, batch_size=1)
        decoder_ids = _dummy_decoder_ids(config, batch_size=1, seq_len=5)

        logits = model(audio, decoder_ids)
        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self):
        """Model should work with different batch sizes."""
        config = _small_config()
        model = WhisperModel(config)
        model.eval()

        for batch_size in [1, 2, 4]:
            audio = _dummy_audio(config, batch_size=batch_size)
            decoder_ids = _dummy_decoder_ids(config, batch_size=batch_size, seq_len=5)
            logits = model(audio, decoder_ids)
            assert logits.shape == (batch_size, 5, config.vocab_size)

    def test_inference_mode(self):
        """Model in inference mode should produce correct output shape."""
        config = _small_config()
        model = WhisperModel(config)
        model.eval()

        assert not model.training
        audio = _dummy_audio(config, batch_size=1)
        decoder_ids = _dummy_decoder_ids(config, batch_size=1, seq_len=5)
        logits = model(audio, decoder_ids)
        assert logits.shape == (1, 5, config.vocab_size)

    def test_config_defaults(self):
        """WhisperConfig should have sensible defaults."""
        config = WhisperConfig()
        assert config.sample_rate == 16000
        assert config.d_model == 64
        assert config.num_encoder_layers == 2
        assert config.num_decoder_layers == 2
        assert config.num_heads == 4
        assert config.dim_feedforward == 256
        assert config.vocab_size == 256
        assert config.dropout == 0.0

    def test_parameter_count_reasonable(self):
        """Small config should have a reasonable parameter count."""
        config = _small_config()
        model = WhisperModel(config)
        total_params = sum(p.numel() for p in model.parameters())

        assert total_params > 0
        assert total_params < 5_000_000  # small config should be well under 5M

    def test_deterministic_inference(self):
        """In inference mode, same input should produce same output."""
        config = _small_config()
        model = WhisperModel(config)
        model.eval()

        audio = _dummy_audio(config, batch_size=1)
        decoder_ids = _dummy_decoder_ids(config, batch_size=1, seq_len=5)

        logits1 = model(audio, decoder_ids)
        logits2 = model(audio, decoder_ids)

        assert torch.allclose(logits1, logits2)

    def test_encoder_output_shape(self):
        """Encoder output should match expected shape from audio."""
        config = _small_config()
        model = WhisperModel(config)
        model.eval()

        audio = _dummy_audio(config, batch_size=2)
        features = model.feature_extractor(audio)
        encoded = model.encoder(features)

        assert encoded.dim() == 3
        assert encoded.shape[0] == 2
        assert encoded.shape[2] == config.d_model

    def test_transcribe_starts_with_zero(self):
        """Transcribe should start generation with token 0."""
        config = _small_config()
        model = WhisperModel(config)
        audio = _dummy_audio(config, batch_size=1)

        tokens = model.transcribe(audio, max_len=5)
        # First token should be 0 (the seed token)
        assert tokens[0, 0].item() == 0
