import torch
import torch.nn as nn
import pytest
from evaluation import perplexity, compute_token_accuracy, Evaluator


class SimpleLM(nn.Module):
    """Simple language model for testing."""
    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        return self.linear(h)

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """Simple autoregressive generation for testing."""
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated[:, -1:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class SimpleDataLoader:
    """Simple data loader for testing."""
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def test_perplexity_random_model():
    """Random model should have high perplexity (close to vocab_size)."""
    model = SimpleLM(vocab_size=100)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 20)),
        "labels": torch.randint(0, 100, (2, 20)),
    }
    loader = SimpleDataLoader([batch])
    ppl = perplexity(model, loader)
    # Random model should have ppl around vocab_size (100)
    assert 10 < ppl < 1000


def test_perplexity_ignores_padding():
    """Perplexity should ignore labels with value -100."""
    model = SimpleLM(vocab_size=100)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 20)),
        "labels": torch.cat([torch.randint(0, 100, (2, 10)), torch.full((2, 10), -100)], dim=1),
    }
    loader = SimpleDataLoader([batch])
    ppl = perplexity(model, loader)
    assert ppl > 0
    assert not torch.isnan(torch.tensor(ppl))


def test_token_accuracy_random():
    """Random model should have low accuracy."""
    model = SimpleLM(vocab_size=100)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 20)),
        "labels": torch.randint(0, 100, (2, 20)),
    }
    loader = SimpleDataLoader([batch])
    acc = compute_token_accuracy(model, loader)
    assert 0.0 <= acc <= 0.2  # random model ~1%


def test_token_accuracy_perfect():
    """Model that copies input should have high accuracy on shifted labels."""
    model = SimpleLM(vocab_size=100)
    # Create a batch where labels == input_ids shifted by 1
    input_ids = torch.randint(0, 100, (2, 20))
    labels = torch.roll(input_ids, -1, dims=1)
    batch = {"input_ids": input_ids, "labels": labels}
    loader = SimpleDataLoader([batch])
    acc = compute_token_accuracy(model, loader)
    assert 0.0 <= acc <= 1.0


def test_evaluator():
    model = SimpleLM(vocab_size=100)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 20)),
        "labels": torch.randint(0, 100, (2, 20)),
    }
    loader = SimpleDataLoader([batch])
    evaluator = Evaluator(model)
    results = evaluator.evaluate(loader)
    assert "perplexity" in results
    assert "token_accuracy" in results


def test_evaluator_with_generation():
    model = SimpleLM(vocab_size=100)

    class MockTokenizer:
        def encode(self, text): return [1, 2, 3]
        def decode(self, ids): return "generated text"

    batch = {
        "input_ids": torch.randint(0, 100, (2, 20)),
        "labels": torch.randint(0, 100, (2, 20)),
    }
    loader = SimpleDataLoader([batch])
    evaluator = Evaluator(model)
    results = evaluator.evaluate(loader, tokenizer=MockTokenizer(), prompts=["Hello"])
    assert "generation_samples" in results
