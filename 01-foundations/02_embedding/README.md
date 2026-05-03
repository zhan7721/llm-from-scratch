# Token Embedding and Rotary Positional Embedding (RoPE)

> **Module 01 — Foundations, Chapter 02**

A neural network processes numbers, not tokens. After tokenization converts text into integer IDs, we need a way to represent those IDs as meaningful vectors. **Embeddings** are that bridge. In this chapter we build two fundamental embedding layers from scratch:

1. **Token Embedding** — a lookup table that converts token IDs to dense vectors
2. **Rotary Positional Embedding (RoPE)** — a rotation-based method that encodes position information

These are the first two layers in any transformer model. Every token passes through them before reaching the attention layers.

---

## Prerequisites

- Basic Python and PyTorch (tensors, `nn.Module`)
- Understanding of what tokenization produces (integer IDs)
- Optional: linear algebra basics (vectors, dot products, rotation matrices)

## Files

| File | Purpose |
|------|---------|
| `embedding.py` | Core implementation of TokenEmbedding and RotaryPositionalEmbedding |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## Why Embeddings?

After tokenization, a sentence like `"The cat sat"` becomes a sequence of integers:

```
[464, 3797, 3332]
```

But neural networks operate on continuous vectors, not discrete integers. We need a function that maps each integer to a vector in a high-dimensional space. This is exactly what an embedding does.

An embedding is a lookup table. It stores one learned vector for each token in the vocabulary. When you pass in token ID 464, it returns the vector stored at row 464.

```
Token ID 464  →  [0.12, -0.45, 0.78, ..., 0.33]  (d_model dimensions)
Token ID 3797 →  [-0.23, 0.56, 0.01, ..., -0.89]
```

These vectors are learned during training. Similar tokens end up with similar vectors, capturing semantic relationships.

---

## Token Embedding

### The Lookup Table

The simplest embedding is `nn.Embedding(vocab_size, d_model)`. It creates a matrix of shape `(vocab_size, d_model)` filled with random values. During training, these values are updated by backpropagation.

```python
self.embedding = nn.Embedding(vocab_size, d_model)
```

When you pass in a tensor of token IDs, it simply indexes into this matrix:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.embedding(x)  # shape: (batch, seq_len, d_model)
```

### Scaling by sqrt(d_model)

In the original Transformer paper (Vaswani et al., 2017), embeddings are scaled by the square root of the model dimension:

```python
return self.embedding(x) * math.sqrt(self.d_model)
```

Why? Without scaling, the magnitude of the embeddings depends on `d_model`. Larger dimensions mean larger dot products in the attention mechanism, which makes the softmax distribution sharper (more peaked). Scaling by `sqrt(d_model)` keeps the dot products in a consistent range regardless of the embedding dimension.

This is a simple but important detail. Many implementations omit it, but it matters for training stability.

---

## Positional Encoding

### The Problem: Permutation Invariance

Transformers process all tokens in parallel. Unlike RNNs, they have no inherent sense of order. Without positional information, the model sees `"The cat sat"` and `"sat cat The"` as identical — just a bag of tokens.

We need to inject position information somehow. There are two main approaches:

1. **Absolute positional encoding**: Add a unique vector to each position's embedding.
2. **Relative positional encoding**: Encode the distance between pairs of positions.

### Absolute Positional Encoding (Sinusoidal)

The original Transformer used sinusoidal functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each position gets a unique pattern of sines and cosines. These patterns are added to the token embeddings before the first attention layer.

**Pros:** Simple, no learned parameters, works for any sequence length.
**Cons:** Added to embeddings, so position information mixes with token information in the first layer.

### Relative Positional Encoding

Relative approaches encode the distance between positions rather than absolute position. The intuition: what matters is not "this is position 5" but "this token is 3 positions away from that token."

**RoPE** is a relative positional encoding that works by rotating vectors. It is used in LLaMA, Mistral, Qwen, and many modern LLMs.

---

## Rotary Positional Embedding (RoPE)

### Intuition: Rotating Vectors

Imagine you have a vector in 2D. You can represent its direction as an angle from the x-axis. Now imagine you have two vectors at different positions. The angle between them encodes their relative position.

RoPE generalizes this idea to high dimensions. For a vector of dimension `d_model`, we treat it as `d_model/2` independent 2D vectors (pairs of dimensions). Each pair is rotated by an angle that depends on the position.

```
Position 0: rotate by 0 * theta
Position 1: rotate by 1 * theta
Position 2: rotate by 2 * theta
...
```

When you compute the dot product between two rotated vectors, the result depends only on the relative position (the difference in rotation angles), not the absolute positions.

### The Mathematics

#### Step 1: Compute Frequencies

Each dimension pair has a different rotation frequency:

```
theta_i = 1 / (base^(2i/d_model))    for i = 0, 1, ..., d_model/2 - 1
```

where `base` is typically 10000.0. Lower dimension pairs rotate faster (higher frequency), higher pairs rotate slower (lower frequency).

```python
inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
```

#### Step 2: Compute Rotation Angles

For each position `m` and dimension pair `i`, the rotation angle is:

```
angle(m, i) = m * theta_i
```

This is computed as an outer product:

```python
t = torch.arange(seq_len)           # positions: [0, 1, 2, ...]
freqs = torch.outer(t, inv_freq)    # shape: (seq_len, d_model/2)
```

#### Step 3: Duplicate and Compute Trig Functions

Duplicate the frequencies to match all `d_model` dimensions:

```python
emb = torch.cat([freqs, freqs], dim=-1)  # shape: (seq_len, d_model)
cos = emb.cos()
sin = emb.sin()
```

#### Step 4: Apply the Rotation

For each vector `x` at position `m`:

```
x_rotated = x * cos(m * theta) + rotate_half(x) * sin(m * theta)
```

where `rotate_half(x)` swaps and negates pairs:

```
rotate_half([x1, x2, x3, x4]) = [-x2, x1, -x4, x3]
```

This is equivalent to applying a 2D rotation matrix to each pair:

```
[cos(theta)  -sin(theta)] [x1]
[sin(theta)   cos(theta)] [x2]
```

### Why RoPE Works

The key property: when you compute the dot product between two RoPE-rotated vectors at positions `m` and `n`, the result depends only on `m - n` (the relative position), not on `m` and `n` individually.

This means:
- The model can attend to relative positions naturally
- No learned positional parameters are needed
- The model can generalize to longer sequences than seen during training

### Code Walkthrough

Here is the complete `RotaryPositionalEmbedding` class:

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.d_model = d_model
        # 1. Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _build_cache(self, seq_len, device):
        # 2. Compute angles for all positions
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_model/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_model)
        return emb.cos(), emb.sin()

    def _rotate_half(self, x):
        # 3. Swap and negate: [x1, x2] -> [-x2, x1]
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        seq_len = x.shape[1]
        cos, sin = self._build_cache(seq_len, x.device)
        cos = cos[:seq_len].unsqueeze(0)  # add batch dim
        sin = sin[:seq_len].unsqueeze(0)
        # 4. Apply rotation
        return x * cos + self._rotate_half(x) * sin
```

**Key observations:**
- `_build_cache` is called on every forward pass in this simple implementation. In production, you would cache the cos/sin tensors.
- `_rotate_half` implements the 90-degree rotation for each dimension pair.
- The forward method applies the rotation formula: `x * cos + rotate(x) * sin`.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/02_embedding/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 01-foundations/02_embedding/tests.py -v -k "test"
```

---

## Exercises

Open `exercise.py` to practice implementing embeddings yourself.

### Exercise Order

1. **`TokenEmbeddingExercise.__init__`** — Create the embedding lookup table
2. **`TokenEmbeddingExercise.forward`** — Look up embeddings and scale by sqrt(d_model)
3. **`RotaryPositionalEmbeddingExercise.__init__`** — Compute inverse frequencies
4. **`RotaryPositionalEmbeddingExercise._build_cache`** — Precompute cos/sin values
5. **`RotaryPositionalEmbeddingExercise._rotate_half`** — Implement the 90-degree rotation
6. **`RotaryPositionalEmbeddingExercise.forward`** — Apply the rotation formula

### Tips

- Start with `TokenEmbedding`. It is just a lookup table with scaling.
- For RoPE, focus on understanding `_rotate_half` first. It is the core operation.
- The `_build_cache` method precomputes all the rotation angles. Think of it as a table of cos/sin values indexed by position and dimension.
- The forward method combines everything: `x * cos + rotate_half(x) * sin`.

---

## Key Takeaways

1. **Embeddings convert discrete tokens to continuous vectors.** The embedding layer is a learned lookup table mapping token IDs to dense vectors.

2. **Scaling by sqrt(d_model) stabilizes training.** It keeps the magnitude of embeddings consistent regardless of the model dimension.

3. **Positional information is essential for transformers.** Without it, the model cannot distinguish between different orderings of the same tokens.

4. **RoPE encodes position through rotation.** Each dimension pair is rotated by an angle proportional to the position. The dot product between two rotated vectors depends only on their relative position.

5. **RoPE is relative, not absolute.** Unlike sinusoidal positional encoding (which is added to embeddings), RoPE is applied in the attention mechanism and naturally captures relative distances.

---

## Further Reading

- [RoPE Paper (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — Original paper introducing Rotary Position Embedding
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Original Transformer paper with sinusoidal positional encoding
- [LLaMA Paper (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) — Uses RoPE in a modern LLM
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual explanation of embeddings and positional encoding
