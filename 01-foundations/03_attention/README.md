# Multi-Head Attention and Grouped Query Attention

> **Module 01 — Foundations, Chapter 03**

Attention is the core mechanism that makes transformers work. It allows each token to look at every other token in the sequence and decide how much to "attend" to each one. In this chapter we build two attention variants from scratch:

1. **Multi-Head Attention (MHA)** — the standard attention mechanism from "Attention Is All You Need"
2. **Grouped Query Attention (GQA)** — an efficient variant used in LLaMA 2/3, Mistral, and other modern LLMs

These are the layers that do the actual "thinking" in a transformer. Everything else — embeddings, layer norms, feed-forward networks — exists to support them.

---

## Prerequisites

- Understanding of embeddings (Chapter 02)
- Linear algebra basics: matrix multiplication, transpose, softmax
- Basic PyTorch: `nn.Linear`, tensor reshaping

## Files

| File | Purpose |
|------|---------|
| `attention.py` | Core implementation of MultiHeadAttention and GroupedQueryAttention |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Intuition

Imagine you are reading the sentence: "The cat, which was very old, sat on the mat."

When you read "sat", you need to know *who* sat. The answer is "cat", which is several words away. Attention gives the model a mechanism to connect "sat" to "cat" directly, regardless of the distance between them.

More formally, attention lets each token compute a weighted sum of all other tokens in the sequence, where the weights are learned based on how relevant each token is.

---

## Scaled Dot-Product Attention

### The Formula

The core operation is:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What information do I provide?"
- **d_k**: Dimension of the key vectors

### Step by Step

#### Step 1: Compute Q, K, V

Each input token vector `x` is projected into three different vectors:

```
Q = x @ W_q    (what this token is looking for)
K = x @ W_k    (what this token offers as a key)
V = x @ W_v    (what information this token provides)
```

These are learned linear projections. The same weights are applied to every position.

#### Step 2: Compute Attention Scores

The score between query at position i and key at position j is their dot product:

```
score(i, j) = Q[i] @ K[j]^T
```

We compute all scores at once as a matrix multiplication:

```
scores = Q @ K^T    # shape: (seq_len, seq_len)
```

Each row i tells us how much position i should attend to every other position.

#### Step 3: Scale

We divide by `sqrt(d_k)`:

```
scores = scores / sqrt(d_k)
```

**Why scale?** Without scaling, as `d_k` grows larger, the dot products grow larger in magnitude. Large values push the softmax into regions with extremely small gradients, making training unstable. Dividing by `sqrt(d_k)` keeps the variance of the dot products at approximately 1, regardless of `d_k`.

This is a mathematical result: if Q and K have independent entries with zero mean and unit variance, then `Q @ K^T` has variance `d_k`. Dividing by `sqrt(d_k)` normalizes the variance back to 1.

#### Step 4: Apply Softmax

Convert scores to probabilities:

```
weights = softmax(scores, dim=-1)    # each row sums to 1
```

Now `weights[i, j]` represents how much position i attends to position j.

#### Step 5: Weighted Sum of Values

Multiply attention weights by V to get the output:

```
output = weights @ V    # shape: (seq_len, d_k)
```

Each output position is a weighted combination of all value vectors, with weights determined by the attention scores.

### Code

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

Three lines. That is the core of attention.

---

## Multi-Head Attention

### Why Multiple Heads?

A single attention head can only learn one type of relationship at a time. With multiple heads, the model can attend to different types of information simultaneously:

- One head might learn syntactic relationships (subject-verb)
- Another might learn semantic similarity (cat-animal)
- Another might learn positional patterns (nearby words)

### How It Works

1. **Split**: Divide the d_model dimensions into `n_heads` groups, each of dimension `d_k = d_model / n_heads`
2. **Attend**: Run scaled dot-product attention independently in each head
3. **Concatenate**: Concatenate the outputs from all heads
4. **Project**: Apply a final linear projection to combine the heads

```python
# Split into heads
Q = Q.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
K = K.view(B, T, n_heads, d_k).transpose(1, 2)
V = V.view(B, T, n_heads, d_k).transpose(1, 2)

# Attend in each head (batched matmul does all heads at once)
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
weights = softmax(scores)
output = weights @ V  # (B, n_heads, T, d_k)

# Concatenate and project
output = output.transpose(1, 2).contiguous().view(B, T, d_model)
output = output @ W_o
```

### Parameter Count

Each head has three projection matrices of size `(d_model, d_k)`. With `n_heads` heads, the total parameter count for Q, K, V is:

```
3 * d_model * d_model = 3 * d_model^2
```

This is the same as having a single large projection — the "split into heads" is just a reshaping trick. The actual parameters are the full `W_q`, `W_k`, `W_v` matrices of shape `(d_model, d_model)`.

---

## Causal Masking

### The Problem

In autoregressive models (like GPT), we generate text one token at a time. When predicting the next token, the model should only see previous tokens, not future ones.

During training, we process the entire sequence at once for efficiency. But we need to ensure that position i cannot attend to positions > i.

### The Solution

Before applying softmax, we set all future positions to negative infinity:

```python
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float("-inf"))
```

After softmax, `exp(-inf) = 0`, so those positions contribute nothing to the attention output.

```
scores = [[ 0.5,  -inf,  -inf]
          [ 0.3,   0.7,  -inf]
          [ 0.2,   0.4,   0.9]]

weights = [[1.0,  0.0,  0.0]
           [0.3,  0.7,  0.0]
           [0.1,  0.2,  0.7]]
```

Position 0 can only attend to itself. Position 1 can attend to positions 0 and 1. Position 2 can attend to all positions. This creates a lower-triangular attention pattern.

---

## Grouped Query Attention (GQA)

### Motivation: The KV Cache Problem

During autoregressive inference, we need to store the K and V vectors for all previous tokens (the "KV cache"). For a model with `n_heads` heads, `d_k` dimension per head, and sequence length T, the cache size per layer is:

```
KV cache = 2 * n_heads * d_k * T * bytes_per_element
```

For a 70B parameter model with 64 heads, 128 dim per head, and 4K context length, this is gigabytes of memory. The KV cache is often the bottleneck for inference throughput.

### The Idea

GQA reduces the number of KV heads. Instead of having `n_heads` KV heads (one per query head), we have `n_kv_heads` KV heads, where `n_kv_heads < n_heads`. Each KV head is shared among a group of query heads.

```
MHA:  8 query heads, 8 KV heads   (each query head has its own KV)
GQA:  8 query heads, 2 KV heads   (4 query heads share each KV head)
MQA:  8 query heads, 1 KV head    (all query heads share one KV)
```

### How It Works

1. **Project Q** to `n_heads` heads (same as MHA)
2. **Project K, V** to `n_kv_heads` heads (fewer than MHA)
3. **Repeat** each KV head `n_rep = n_heads / n_kv_heads` times to match the query head count
4. **Attend** as usual

```python
# Project
Q = W_q(x).view(B, T, n_heads, d_k)       # (B, T, 8, d_k)
K = W_k(x).view(B, T, n_kv_heads, d_k)     # (B, T, 2, d_k)
V = W_v(x).view(B, T, n_kv_heads, d_k)     # (B, T, 2, d_k)

# Repeat KV heads
K = repeat_kv(K)  # (B, T, 2, d_k) -> (B, T, 8, d_k)
V = repeat_kv(V)  # (B, T, 2, d_k) -> (B, T, 8, d_k)

# Standard attention from here
scores = Q @ K^T / sqrt(d_k)
```

### Parameter Savings

With `n_kv_heads` KV heads instead of `n_heads`:

```
MHA KV params: 2 * d_model * d_model
GQA KV params: 2 * d_model * (n_kv_heads / n_heads) * d_model
```

For LLaMA 2 70B: `n_heads=64, n_kv_heads=8`, so KV parameters are reduced by 8x.

### Quality Trade-off

GQA with `n_kv_heads` between 1 (MQA) and `n_heads` (MHA) offers a good balance:

| Method | KV Heads | Quality | KV Cache Size |
|--------|----------|---------|---------------|
| MHA    | n_heads  | Best    | Largest       |
| GQA    | 8        | ~MHA    | 8x smaller    |
| MQA    | 1        | Worse   | Smallest      |

LLaMA 2 70B uses GQA with 8 KV heads and achieves comparable quality to MHA with significant inference speedup.

---

## Comparison: MHA vs MQA vs GQA

### Multi-Head Attention (MHA)
- **KV heads**: `n_heads` (one per query head)
- **Pros**: Best quality, most expressive
- **Cons**: Large KV cache, slow inference
- **Used in**: GPT-3, original Transformer

### Multi-Query Attention (MQA)
- **KV heads**: 1 (shared across all query heads)
- **Pros**: Smallest KV cache, fastest inference
- **Cons**: Quality degradation, especially for large models
- **Used in**: PaLM, Falcon

### Grouped Query Attention (GQA)
- **KV heads**: `n_kv_heads` (a middle ground)
- **Pros**: Near-MHA quality, much smaller KV cache
- **Cons**: Slightly more complex implementation
- **Used in**: LLaMA 2/3, Mistral, Qwen

---

## Code Walkthrough

### MultiHeadAttention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Project and reshape to multi-head format
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # Masks
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention
        weights = F.softmax(scores, dim=-1)
        output = weights @ V

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(output)
```

### GroupedQueryAttention

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, causal=False):
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # fewer!
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # fewer!
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _repeat_kv(self, x):
        """Duplicate KV heads to match query head count."""
        B, n_kv, T, d_k = x.shape
        if self.n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, d_k).reshape(
            B, self.n_heads, T, d_k
        )

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Expand KV heads
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # Same attention from here
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        # ... masks, softmax, output ...
```

**Key difference**: K and V projections output `n_kv_heads * d_k` instead of `n_heads * d_k`. The `_repeat_kv` method expands them back to `n_heads` for the attention computation.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/03_attention/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 01-foundations/03_attention/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing attention yourself.

### Exercise Order

1. **`MultiHeadAttentionExercise.__init__`** — Create W_q, W_k, W_v, W_o linear layers
2. **`MultiHeadAttentionExercise.forward`** — Implement the full MHA forward pass
3. **`GroupedQueryAttentionExercise.__init__`** — Create projections with fewer KV heads
4. **`GroupedQueryAttentionExercise._repeat_kv`** — Implement KV head repetition
5. **`GroupedQueryAttentionExercise.forward`** — Implement the full GQA forward pass

### Tips

- Start with MHA. The forward pass is: project -> reshape -> scores -> mask -> softmax -> output -> project.
- The `.view(B, T, n_heads, d_k).transpose(1, 2)` pattern is how you split a flat vector into multiple heads. Understand why transpose is needed (we want heads as dimension 1 for batched matmul).
- For GQA, the only difference is that K and V have fewer heads. The `_repeat_kv` method bridges the gap.
- The causal mask is just an upper-triangular matrix of bools. `torch.triu` creates it.

---

## Key Takeaways

1. **Attention computes weighted sums based on relevance.** Each token looks at all other tokens and decides how much to attend to each one.

2. **Scaling by sqrt(d_k) stabilizes training.** Without it, large dot products cause softmax to saturate, leading to vanishing gradients.

3. **Multiple heads capture different relationships.** Each head can learn to attend to different aspects of the input (syntax, semantics, position).

4. **Causal masking enables autoregressive generation.** By blocking future positions, the model can only use past context when making predictions.

5. **GQA reduces KV cache size with minimal quality loss.** By sharing KV heads among groups of query heads, we get significant inference speedup for a small quality trade-off.

---

## Further Reading

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [GQA Paper (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) — Grouped Query Attention
- [LLaMA 2 Paper (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — Uses GQA in a production model
- [Multi-Query Attention (Shazeer, 2019)](https://arxiv.org/abs/1911.02150) — Original MQA proposal
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual explanation of attention
