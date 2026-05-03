import torch
import pytest
from kv_cache import CachedAttention, KVCache


def test_kv_cache_stores_values():
    cache = KVCache(max_batch_size=2, max_seq_len=10, n_heads=4, d_k=16)
    k = torch.randn(2, 4, 5, 16)
    v = torch.randn(2, 4, 5, 16)
    cache.update(2, 5, k, v)
    assert cache.k_cache.shape == (2, 4, 10, 16)


def test_cached_attention_autoregressive():
    attn = CachedAttention(d_model=64, n_heads=4)
    cache = KVCache(max_batch_size=1, max_seq_len=20, n_heads=4, d_k=16)

    x1 = torch.randn(1, 1, 64)
    out1 = attn(x1, cache, start_pos=0)

    x2 = torch.randn(1, 1, 64)
    out2 = attn(x2, cache, start_pos=1)

    assert out1.shape == (1, 1, 64)
    assert out2.shape == (1, 1, 64)


def test_cached_attention_without_cache():
    attn = CachedAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.shape == (2, 10, 64)


def test_kv_cache_get():
    cache = KVCache(max_batch_size=2, max_seq_len=10, n_heads=4, d_k=16)
    k = torch.randn(2, 4, 5, 16)
    v = torch.randn(2, 4, 5, 16)
    cache.update(2, 0, k, v)
    k_out, v_out = cache.get(1, 5)
    assert k_out.shape == (1, 4, 5, 16)
    assert torch.allclose(k_out, k[:1])


def test_cached_vs_uncached_equivalence():
    """Cached and uncached attention should produce the same output."""
    torch.manual_seed(42)
    attn = CachedAttention(d_model=64, n_heads=4)
    x = torch.randn(1, 5, 64)

    # Uncached
    out_uncached = attn(x)

    # Cached (token by token)
    cache = KVCache(max_batch_size=1, max_seq_len=10, n_heads=4, d_k=16)
    out_cached = []
    for i in range(5):
        token_out = attn(x[:, i:i+1, :], cache, start_pos=i)
        out_cached.append(token_out)
    out_cached = torch.cat(out_cached, dim=1)

    assert torch.allclose(out_uncached, out_cached, atol=1e-5)
