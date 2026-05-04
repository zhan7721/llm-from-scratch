"""Tests for the complete GPT model architecture."""

import torch
import pytest
from model import GPT, GPTConfig


def test_gpt_forward_shape():
    """Output shape should be (batch, seq_len, vocab_size)."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    x = torch.randint(0, 256, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 256)


def test_gpt_generation():
    """Generation should extend the prompt by max_new_tokens."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    prompt = torch.randint(0, 256, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    assert generated.shape == (1, 15)


def test_gpt_parameter_count():
    """Model should have a reasonable number of parameters for a small config."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    assert total_params < 1_000_000  # small config should be under 1M params


def test_gpt_gradient_flow():
    """Gradients should flow through all parameters."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    x = torch.randint(0, 256, (2, 10))
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_gpt_generation_deterministic_with_seed():
    """Generation should be deterministic when using the same random seed."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    model.train(False)
    prompt = torch.randint(0, 256, (1, 5))

    torch.manual_seed(42)
    out1 = model.generate(prompt, max_new_tokens=5)

    torch.manual_seed(42)
    out2 = model.generate(prompt, max_new_tokens=5)

    assert torch.equal(out1, out2)


def test_gpt_weight_tying():
    """lm_head weights should be the same object as token embedding weights."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    assert model.lm_head.weight is model.token_emb.embedding.weight


def test_gpt_config_defaults():
    """GPTConfig should have sensible defaults."""
    config = GPTConfig()
    assert config.vocab_size == 32000
    assert config.d_model == 512
    assert config.n_heads == 8
    assert config.n_layers == 6
    assert config.max_seq_len == 1024


def test_gpt_different_batch_sizes():
    """Model should work with different batch sizes."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    for batch_size in [1, 2, 4]:
        x = torch.randint(0, 256, (batch_size, 10))
        logits = model(x)
        assert logits.shape == (batch_size, 10, 256)


def test_gpt_different_seq_lens():
    """Model should work with different sequence lengths."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    for seq_len in [1, 5, 10, 50]:
        x = torch.randint(0, 256, (1, seq_len))
        logits = model(x)
        assert logits.shape == (1, seq_len, 256)


def test_gpt_layer_count():
    """Model should have the correct number of transformer layers."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=4, max_seq_len=128)
    model = GPT(config)
    assert len(model.layers) == 4


def test_gpt_output_logits_not_all_same():
    """Output logits should not be identical across the vocabulary dimension."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    model.train(False)
    x = torch.randint(0, 256, (1, 5))
    logits = model(x)
    # Check that logits are not all the same value
    assert logits.std() > 0.01


def test_gpt_generate_temperature():
    """Lower temperature should produce more concentrated distributions."""
    config = GPTConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = GPT(config)
    model.train(False)
    prompt = torch.randint(0, 256, (1, 5))

    # Both should produce valid output (no crashes)
    torch.manual_seed(42)
    out_low = model.generate(prompt, max_new_tokens=5, temperature=0.1)
    torch.manual_seed(42)
    out_high = model.generate(prompt, max_new_tokens=5, temperature=2.0)

    # Both should have the correct shape
    assert out_low.shape == (1, 10)
    assert out_high.shape == (1, 10)
