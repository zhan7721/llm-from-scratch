# KV Cache for Efficient Autoregressive Generation

## The Problem: Naive Generation is Wasteful

When generating text with a transformer model, we typically do it **autoregressively** — one token at a time. The naive approach looks like this:

```
Step 1: Generate token 1 from input [prompt]
Step 2: Generate token 2 from input [prompt, token1]
Step 3: Generate token 3 from input [prompt, token1, token2]
...
```

At each step, we run the **full attention computation** over the entire sequence so far. This means:
- Step 1: O(1) attention
- Step 2: O(2) attention
- Step 3: O(3) attention
- ...
- Step n: O(n) attention

**Total cost: O(n²)** — and most of this work is **redundant**! At step 3, we've already computed the Key and Value vectors for positions 1 and 2 at step 2. Why recompute them?

## The Solution: KV Cache

The **KV Cache** stores the Key and Value tensors from previous steps, so we only need to compute the **new** K and V for the current token.

### How It Works

1. **Pre-allocate** a cache buffer large enough for the maximum sequence length
2. At each generation step:
   - Compute Q, K, V for the **current token only** (seq_len = 1)
   - **Update** the cache with the new K, V at the current position
   - **Retrieve** the full K, V history from the cache
   - Compute attention: new Q against all cached K, V

```python
# Without cache (naive):
for i in range(seq_len):
    # Recompute ALL K, V from scratch
    q, k, v = attention(full_sequence[:i+1])

# With cache (efficient):
cache = KVCache()
for i in range(seq_len):
    # Only compute NEW K, V
    q, k_new, v_new = attention(current_token)
    cache.update(i, k_new, v_new)
    k_all, v_all = cache.get(i + 1)
    output = attend(q, k_all, v_all)
```

### Complexity Comparison

| Approach | Per-Step Cost | Total Cost (n tokens) | Memory |
|----------|---------------|----------------------|--------|
| Naive   | O(i)         | O(n²)               | O(1)   |
| KV Cache | O(1)        | O(n)                | O(n)   |

The KV Cache trades **memory** for **compute** — we store O(n) cache entries but avoid O(n²) recomputation.

## Implementation Details

### KVCache Class

```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_heads, d_k):
        # Pre-allocate buffers
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
```

Key design decisions:
- **Pre-allocation**: We allocate the full buffer upfront to avoid repeated memory allocation
- **In-place updates**: We write directly into the pre-allocated buffer
- **Batch-aware**: The cache handles multiple sequences efficiently

### CachedAttention Class

The attention module needs to handle two modes:
1. **Prefill mode**: Processing the initial prompt (seq_len > 1), no cache needed
2. **Generation mode**: Generating one token at a time (seq_len = 1), uses cache

```python
def forward(self, x, kv_cache=None, start_pos=0):
    B, T, _ = x.shape

    # Always compute Q, K, V for current input
    q = self.W_q(x).view(B, T, n_heads, d_k).transpose(1, 2)
    k = self.W_k(x).view(B, T, n_heads, d_k).transpose(1, 2)
    v = self.W_v(x).view(B, T, n_heads, d_k).transpose(1, 2)

    if kv_cache is not None:
        # Store new K, V and retrieve full history
        kv_cache.update(B, start_pos, k, v)
        k, v = kv_cache.get(B, start_pos + T)

    # Standard attention computation
    scores = q @ k.T / sqrt(d_k)
    # Apply causal mask if needed
    output = softmax(scores) @ v
    return self.W_o(output)
```

## Multi-Batch Considerations

When processing multiple sequences in a batch:
- All sequences share the same cache buffer
- Each sequence uses positions `[0:actual_length]`
- Padding sequences to the same length wastes cache space
- Advanced: use variable-length packing for efficiency

```python
# Batch processing
cache = KVCache(max_batch_size=4, max_seq_len=100, n_heads=8, d_k=64)

# Process 4 sequences simultaneously
for pos in range(max_len):
    token = get_next_tokens(batch)  # (4, 1, d_model)
    output = attention(token, cache, start_pos=pos)
```

## Memory Analysis

For a typical transformer layer:
- **Cache size per layer**: 2 × batch_size × n_heads × seq_len × d_k
- **Total cache**: num_layers × cache_per_layer

Example (GPT-2 small):
- 12 layers, 12 heads, d_k = 64, seq_len = 1024, batch = 1
- Cache per layer: 2 × 1 × 12 × 1024 × 64 = 1.5 MB
- Total cache: 12 × 1.5 MB = **18 MB**

For longer sequences or larger models, this can become significant!

## Performance Comparison

Benchmarking with vs without cache:

```python
# Without cache: ~O(n²) time
start = time.time()
for i in range(1000):
    output = model(full_sequence[:i+1])
naive_time = time.time() - start

# With cache: ~O(n) time
start = time.time()
cache = KVCache()
for i in range(1000):
    output = model(current_token, cache, start_pos=i)
cached_time = time.time() - start

print(f"Speedup: {naive_time / cached_time:.1f}x")
# Typical speedup: 10-100x for long sequences
```

## Code Walkthrough

### Step 1: Initialize Cache
```python
cache = KVCache(max_batch_size=1, max_seq_len=2048, n_heads=12, d_k=64)
```

### Step 2: Prefill (Process Prompt)
```python
prompt = tokenize("Once upon a time")
output = model(prompt, cache, start_pos=0)
# Cache now contains K, V for positions 0..3
```

### Step 3: Generate Tokens
```python
for i in range(100):
    next_token = sample(output)
    output = model(next_token, cache, start_pos=len(prompt) + i)
    # Cache grows by 1 position each step
```

## Key Takeaways

1. **KV Cache eliminates redundant computation** by storing previous K, V
2. **Trade memory for speed** — O(n) cache vs O(n²) recomputation
3. **Pre-allocate buffers** to avoid allocation overhead during generation
4. **Handle prefill and generation modes** differently in attention
5. **Essential for production inference** — all modern LLM serving systems use KV cache

## Next Steps

- **PagedAttention**: More memory-efficient cache management (used in vLLM)
- **Multi-Query Attention (MQA)**: Share K, V across heads to reduce cache size
- **Grouped-Query Attention (GQA)**: Compromise between MHA and MQA
- **Quantized KV Cache**: Store cache in lower precision (int8, int4)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — Analysis of KV cache trade-offs
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention implementation
