import pytest
from tokenizer import BPETokenizer


def test_train_basic():
    """Tokenizer should learn merges from a corpus."""
    corpus = "low lower newest wide"
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)
    assert len(tokenizer.vocab) > 256  # base byte vocab + learned merges


def test_encode_decode_roundtrip():
    """Encoding then decoding should return the original text."""
    corpus = "hello world " * 100
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_encode_returns_list_of_ints():
    """encode() should return a list of integer token IDs."""
    corpus = "abc def " * 50
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)
    ids = tokenizer.encode("abc")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_unknown_bytes_fallback():
    """Tokenizer should handle bytes not seen during training."""
    corpus = "aaa " * 50
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)
    ids = tokenizer.encode("xyz")
    assert len(ids) > 0


def test_vocab_size_respected():
    """Final vocab size should not exceed the specified limit."""
    corpus = "abcdefghij " * 100
    tokenizer = BPETokenizer(vocab_size=270)
    tokenizer.train(corpus)
    assert len(tokenizer.vocab) <= 270
