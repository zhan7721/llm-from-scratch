"""Tests for the Text-to-Speech generation implementation."""

import torch
from speech_generation import (
    TextEncoder,
    Vocoder,
    SimpleTTS,
    TTSLoss,
    TTSConfig,
)


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------
def _small_config(**overrides) -> TTSConfig:
    """Create a small TTSConfig suitable for unit testing."""
    defaults = dict(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        max_text_positions=128,
        n_mels=40,
        vocoder_channels=32,
        vocoder_num_layers=2,
        hop_length=128,
        dropout=0.0,
    )
    defaults.update(overrides)
    return TTSConfig(**defaults)


def _dummy_text_ids(config: TTSConfig, batch_size: int = 2, seq_len: int = 10) -> torch.Tensor:
    """Create dummy text token IDs."""
    return torch.randint(0, config.vocab_size, (batch_size, seq_len))


def _dummy_mel(config: TTSConfig, batch_size: int = 2, num_frames: int = 30) -> torch.Tensor:
    """Create dummy mel spectrogram tensors."""
    return torch.randn(batch_size, config.n_mels, num_frames)


# ===========================================================================
# TextEncoder tests
# ===========================================================================

class TestTextEncoder:
    """Tests for the TextEncoder module."""

    def test_output_shape(self):
        """TextEncoder output should be (B, seq_len, d_model)."""
        config = _small_config()
        encoder = TextEncoder(config)
        text_ids = _dummy_text_ids(config, batch_size=2, seq_len=10)
        out = encoder(text_ids)

        assert out.dim() == 3
        assert out.shape == (2, 10, config.d_model)

    def test_output_shape_batch_1(self):
        """TextEncoder should work with batch_size=1."""
        config = _small_config()
        encoder = TextEncoder(config)
        text_ids = _dummy_text_ids(config, batch_size=1, seq_len=5)
        out = encoder(text_ids)

        assert out.shape == (1, 5, config.d_model)

    def test_preserves_sequence_length(self):
        """TextEncoder should not change sequence length."""
        config = _small_config()
        encoder = TextEncoder(config)

        for seq_len in [1, 5, 20]:
            text_ids = _dummy_text_ids(config, batch_size=1, seq_len=seq_len)
            out = encoder(text_ids)
            assert out.shape[1] == seq_len

    def test_gradient_flows(self):
        """Gradients should flow through the TextEncoder."""
        config = _small_config()
        encoder = TextEncoder(config)
        text_ids = _dummy_text_ids(config, batch_size=1, seq_len=5)
        out = encoder(text_ids)
        loss = out.sum()
        loss.backward()

        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self):
        """TextEncoder should work with different batch sizes."""
        config = _small_config()
        encoder = TextEncoder(config)
        encoder.eval()

        for batch_size in [1, 2, 4]:
            text_ids = _dummy_text_ids(config, batch_size=batch_size, seq_len=5)
            out = encoder(text_ids)
            assert out.shape == (batch_size, 5, config.d_model)

    def test_num_layers(self):
        """TextEncoder should have the correct number of layers."""
        config = _small_config(num_layers=4)
        encoder = TextEncoder(config)
        assert len(encoder.layers) == 4


# ===========================================================================
# Vocoder tests
# ===========================================================================

class TestVocoder:
    """Tests for the Vocoder module."""

    def test_output_shape(self):
        """Vocoder output should be (B, num_samples)."""
        config = _small_config()
        vocoder = Vocoder(config)
        mel = _dummy_mel(config, batch_size=2, num_frames=20)
        waveform = vocoder(mel)

        assert waveform.dim() == 2
        assert waveform.shape[0] == 2
        # num_samples = num_frames * hop_length
        expected_samples = 20 * config.hop_length
        assert waveform.shape[1] == expected_samples

    def test_output_shape_batch_1(self):
        """Vocoder should work with batch_size=1."""
        config = _small_config()
        vocoder = Vocoder(config)
        mel = _dummy_mel(config, batch_size=1, num_frames=10)
        waveform = vocoder(mel)

        assert waveform.shape[0] == 1
        assert waveform.shape[1] == 10 * config.hop_length

    def test_gradient_flows(self):
        """Gradients should flow through the Vocoder."""
        config = _small_config()
        vocoder = Vocoder(config)
        mel = _dummy_mel(config, batch_size=1, num_frames=10)
        waveform = vocoder(mel)
        loss = waveform.sum()
        loss.backward()

        for name, p in vocoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_different_num_frames(self):
        """Vocoder should handle different numbers of mel frames."""
        config = _small_config()
        vocoder = Vocoder(config)
        vocoder.eval()

        for num_frames in [5, 10, 20]:
            mel = _dummy_mel(config, batch_size=1, num_frames=num_frames)
            waveform = vocoder(mel)
            assert waveform.shape[1] == num_frames * config.hop_length

    def test_different_batch_sizes(self):
        """Vocoder should work with different batch sizes."""
        config = _small_config()
        vocoder = Vocoder(config)
        vocoder.eval()

        for batch_size in [1, 2, 4]:
            mel = _dummy_mel(config, batch_size=batch_size, num_frames=10)
            waveform = vocoder(mel)
            assert waveform.shape == (batch_size, 10 * config.hop_length)


# ===========================================================================
# SimpleTTS tests
# ===========================================================================

class TestSimpleTTS:
    """Tests for the full SimpleTTS model."""

    def test_forward_shape(self):
        """Forward pass should produce (B, n_mels, num_frames) mel spectrogram."""
        config = _small_config()
        model = SimpleTTS(config)
        text_ids = _dummy_text_ids(config, batch_size=2, seq_len=10)
        target_mel = _dummy_mel(config, batch_size=2, num_frames=30)

        predicted_mel = model(text_ids, target_mel)
        assert predicted_mel.shape == (2, config.n_mels, 30)

    def test_forward_without_target(self):
        """Forward pass without target_mel should produce valid output."""
        config = _small_config()
        model = SimpleTTS(config)
        model.eval()
        text_ids = _dummy_text_ids(config, batch_size=1, seq_len=5)

        predicted_mel = model(text_ids, target_mel=None)
        assert predicted_mel.dim() == 3
        assert predicted_mel.shape[0] == 1
        assert predicted_mel.shape[1] == config.n_mels

    def test_synthesize_produces_waveform(self):
        """Synthesize should produce a 2D waveform tensor."""
        config = _small_config()
        model = SimpleTTS(config)
        text_ids = _dummy_text_ids(config, batch_size=1, seq_len=5)

        # First get predicted mel to know expected num_frames
        model.training = False
        with torch.no_grad():
            predicted_mel = model(text_ids)
        num_frames = predicted_mel.shape[2]

        waveform = model.synthesize(text_ids)
        expected_samples = num_frames * config.hop_length
        assert waveform.dim() == 2
        assert waveform.shape[0] == 1
        assert waveform.shape[1] == expected_samples

    def test_synthesize_batch(self):
        """Synthesize should work with multiple inputs."""
        config = _small_config()
        model = SimpleTTS(config)
        text_ids = _dummy_text_ids(config, batch_size=3, seq_len=5)

        waveform = model.synthesize(text_ids)
        assert waveform.dim() == 2
        assert waveform.shape[0] == 3

    def test_gradient_flows_full_model(self):
        """Gradients should flow through the entire model.

        The duration predictor output is used as a non-differentiable integer
        (num_frames), so its parameters don't receive gradients from the mel
        reconstruction loss. In a real TTS system, it would be trained with
        a separate duration loss. We skip it here.
        """
        config = _small_config()
        model = SimpleTTS(config)
        text_ids = _dummy_text_ids(config, batch_size=1, seq_len=5)
        target_mel = _dummy_mel(config, batch_size=1, num_frames=20)

        predicted_mel = model(text_ids, target_mel)
        loss = predicted_mel.sum()
        loss.backward()

        # Duration predictor: output is converted to a non-differentiable integer
        # Vocoder: only used during inference (synthesize), not in forward
        skip_patterns = {"duration_predictor", "vocoder"}
        for name, p in model.named_parameters():
            if p.requires_grad:
                if any(pat in name for pat in skip_patterns):
                    continue
                assert p.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self):
        """Model should work with different batch sizes."""
        config = _small_config()
        model = SimpleTTS(config)
        model.eval()

        for batch_size in [1, 2, 4]:
            text_ids = _dummy_text_ids(config, batch_size=batch_size, seq_len=5)
            target_mel = _dummy_mel(config, batch_size=batch_size, num_frames=10)
            predicted_mel = model(text_ids, target_mel)
            assert predicted_mel.shape == (batch_size, config.n_mels, 10)

    def test_inference_mode(self):
        """Model in inference mode should produce correct output shape."""
        config = _small_config()
        model = SimpleTTS(config)
        model.eval()

        assert not model.training
        text_ids = _dummy_text_ids(config, batch_size=1, seq_len=5)
        target_mel = _dummy_mel(config, batch_size=1, num_frames=10)
        predicted_mel = model(text_ids, target_mel)
        assert predicted_mel.shape == (1, config.n_mels, 10)

    def test_config_defaults(self):
        """TTSConfig should have sensible defaults."""
        config = TTSConfig()
        assert config.vocab_size == 256
        assert config.d_model == 64
        assert config.num_heads == 4
        assert config.num_layers == 2
        assert config.dim_feedforward == 256
        assert config.max_text_positions == 512
        assert config.n_mels == 80
        assert config.vocoder_channels == 64
        assert config.vocoder_num_layers == 4
        assert config.hop_length == 256
        assert config.dropout == 0.0

    def test_parameter_count_reasonable(self):
        """Small config should have a reasonable parameter count."""
        config = _small_config()
        model = SimpleTTS(config)
        total_params = sum(p.numel() for p in model.parameters())

        assert 1000 < total_params < 500_000

    def test_deterministic_inference(self):
        """In inference mode, same input should produce same output."""
        config = _small_config()
        model = SimpleTTS(config)
        model.eval()

        text_ids = _dummy_text_ids(config, batch_size=1, seq_len=5)
        target_mel = _dummy_mel(config, batch_size=1, num_frames=10)

        mel1 = model(text_ids, target_mel)
        mel2 = model(text_ids, target_mel)

        assert torch.allclose(mel1, mel2)


# ===========================================================================
# TTSLoss tests
# ===========================================================================

class TestTTSLoss:
    """Tests for the TTSLoss module."""

    def test_loss_scalar(self):
        """TTSLoss should produce a scalar loss."""
        config = _small_config()
        loss_fn = TTSLoss()
        predicted = _dummy_mel(config, batch_size=2, num_frames=20)
        target = _dummy_mel(config, batch_size=2, num_frames=20)

        loss = loss_fn(predicted, target)
        assert loss.dim() == 0  # scalar

    def test_loss_non_negative(self):
        """MSE loss should be non-negative."""
        loss_fn = TTSLoss()
        predicted = torch.randn(2, 40, 10)
        target = torch.randn(2, 40, 10)

        loss = loss_fn(predicted, target)
        assert loss.item() >= 0.0

    def test_loss_zero_when_equal(self):
        """MSE loss should be zero when prediction equals target."""
        loss_fn = TTSLoss()
        tensor = torch.randn(2, 40, 10)

        loss = loss_fn(tensor, tensor)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_loss_differentiable(self):
        """TTSLoss should be differentiable."""
        loss_fn = TTSLoss()
        predicted = torch.randn(1, 40, 10, requires_grad=True)
        target = torch.randn(1, 40, 10)

        loss = loss_fn(predicted, target)
        loss.backward()

        assert predicted.grad is not None
