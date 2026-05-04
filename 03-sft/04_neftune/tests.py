import torch
import torch.nn as nn
import pytest
from neftune import NEFTuneEmbedding, NEFTuneConfig, apply_neftune, compute_neftune_noise_scale


def test_neftune_embedding_shape():
    emb = NEFTuneEmbedding(100, 32, alpha=5.0)
    x = torch.randint(0, 100, (2, 10))
    out = emb(x)
    assert out.shape == (2, 10, 32)


def test_neftune_training_noise():
    """Training mode should add noise."""
    emb = NEFTuneEmbedding(100, 32, alpha=5.0)
    emb.train()
    x = torch.randint(0, 100, (1, 10))

    out1 = emb(x)
    out2 = emb(x)

    # Should be different due to random noise (no re-seeding between calls)
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_neftune_eval_no_noise():
    """Eval mode should not add noise."""
    emb = NEFTuneEmbedding(100, 32, alpha=5.0)
    emb.eval()
    x = torch.randint(0, 100, (1, 10))

    out1 = emb(x)
    out2 = emb(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_neftune_alpha_zero():
    """Alpha=0 should be equivalent to standard embedding."""
    emb = NEFTuneEmbedding(100, 32, alpha=0.0)
    emb.train()
    x = torch.randint(0, 100, (1, 10))
    out = emb(x)
    orig = emb.embedding(x)
    assert torch.allclose(out, orig, atol=1e-6)


def test_neftune_noise_scale():
    scale = compute_neftune_noise_scale(seq_len=100, embedding_dim=64, alpha=5.0)
    assert scale > 0
    # Longer sequences should have smaller noise
    scale_short = compute_neftune_noise_scale(seq_len=10, embedding_dim=64, alpha=5.0)
    assert scale_short > scale


def test_neftune_config_constant():
    config = NEFTuneConfig(alpha=5.0, schedule="constant")
    assert config.get_alpha(0) == 5.0
    assert config.get_alpha(100) == 5.0


def test_neftune_config_warmup():
    config = NEFTuneConfig(alpha=5.0, schedule="warmup", warmup_steps=100)
    assert config.get_alpha(0) == 0.0
    assert config.get_alpha(50) == 2.5
    assert config.get_alpha(100) == 5.0


def test_apply_neftune():
    model = nn.Sequential(nn.Embedding(100, 32), nn.Linear(32, 10))
    model = apply_neftune(model, alpha=5.0)
    # Check that embedding was replaced
    assert isinstance(model[0], NEFTuneEmbedding)


def test_neftune_gradient_flow():
    emb = NEFTuneEmbedding(100, 32, alpha=5.0)
    emb.train()
    x = torch.randint(0, 100, (2, 10))
    out = emb(x)
    out.sum().backward()
    assert emb.embedding.weight.grad is not None
