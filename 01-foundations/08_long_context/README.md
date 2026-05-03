# Long Context: RoPE Scaling and YaRN

## The Problem: Fixed Context Length

Most large language models are trained with a **fixed context length** — for example, LLaMA 2 was trained on 4096 tokens, and GPT-4 on 8192 or 32K tokens. This means:

- The model has never "seen" positions beyond its training length
- Attention patterns at unseen positions become unpredictable
- Naively extending the sequence length leads to severe quality degradation

But what if we need to process a 50-page document, or maintain a long conversation? We need techniques to **extend the context window** without retraining the model from scratch.

## Background: Rotary Position Embeddings (RoPE)

Before understanding context extension, let's review how RoPE works (covered in Chapter 02 - Embedding).

RoPE encodes position information by **rotating** query and key vectors in attention. For a position `m` and dimension pair `i`, the rotation angle is:

```
theta_i = m * base^(-2i/d_model)
```

where `base` is typically 10000. The rotation is applied as:

```
[q_2i, q_2i+1] -> [q_2i * cos(m*theta_i) - q_2i+1 * sin(m*theta_i),
                     q_2i * sin(m*theta_i) + q_2i+1 * cos(m*theta_i)]
```

The key insight: **each dimension pair rotates at a different frequency**. Low dimensions rotate fast (encoding local context), while high dimensions rotate slowly (encoding global position).

## Approach 1: Position Interpolation (Linear Scaling)

The simplest approach to extend context: **squash the positions** so they fit within the original training range.

### How It Works

If the model was trained on positions `[0, 4096)` and we want to handle positions up to 8192, we divide all positions by 2:

```
position_scaled = position / scale_factor
```

For scale_factor = 2.0:
- Original position 0 -> scaled 0.0
- Original position 4096 -> scaled 2048.0
- Original position 8192 -> scaled 4096.0

Now position 8192 maps to the same encoding that position 4096 had during training.

### Implementation

```python
class ScaledRoPE(nn.Module):
    def __init__(self, d_model, base=10000.0, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        # Scale positions
        t = torch.arange(seq_len, device=x.device) / self.scale_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin
```

### Trade-offs

**Pros:**
- Simple to implement
- Works well for moderate extensions (2x-4x)
- Used in LLaMA 2 Long with good results

**Cons:**
- Compresses ALL frequencies equally, including high-frequency ones that encode local context
- May lose fine-grained position discrimination for nearby tokens
- Requires some fine-tuning to recover quality

## Approach 2: NTK-Aware Scaling

Instead of scaling positions, NTK-aware scaling **adjusts the base frequency**. This preserves high-frequency components while extending the low-frequency range.

### The Insight

Recall that each dimension pair has a different frequency:

```
theta_i = base^(-2i/d_model)
```

- **Low dimensions** (small i): high frequency, encode local context
- **High dimensions** (large i): low frequency, encode global position

Linear scaling compresses everything. But we only need to extend the **low-frequency** dimensions (global position). The high-frequency dimensions (local context) should stay the same!

### The Formula

NTK-aware scaling modifies the base:

```
base_scaled = base * (scale_factor ^ (d_model / (d_model - 2)))
```

This has the effect of:
- Leaving high-frequency dimensions mostly unchanged
- Extending low-frequency dimensions proportionally to the scale factor

### Why "NTK"?

The name comes from Neural Tangent Kernel theory. The key insight is that different frequency components of the position encoding have different "learnability" characteristics. High-frequency components need to be preserved for local context understanding.

## Approach 3: YaRN (Yet another RoPE extensioN)

YaRN combines NTK-aware scaling with **attention temperature scaling** for even better results.

### Components

1. **NTK-aware scaling** (as described above): Adjusts the base frequency
2. **Attention temperature**: Scales down attention logits to compensate for the changed frequency distribution

```python
# Attention temperature factor
attn_factor = 1 / sqrt(scale_factor)
```

### Why Temperature Scaling?

After NTK-aware scaling, the frequency distribution changes. The attention scores (dot products of Q and K) can become too large, leading to:
- Overconfident attention patterns
- Reduced diversity in attention weights
- Potential numerical instability

The temperature factor `1/sqrt(scale_factor)` brings the attention scores back to a reasonable range.

### Implementation

```python
class YaRNRope(nn.Module):
    def __init__(self, d_model, base=10000.0, scale_factor=4.0):
        super().__init__()
        # NTK-aware base scaling
        base_scaled = base * (scale_factor ** (d_model / (d_model - 2)))
        inv_freq = 1.0 / (base_scaled ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Attention temperature
        self.attn_factor = 1.0 / math.sqrt(scale_factor)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        # Apply rotation AND temperature scaling
        return (x * cos + self._rotate_half(x) * sin) * self.attn_factor
```

## Linear Scaling vs YaRN: When to Use Which

| Feature | Linear Scaling | YaRN |
|---------|---------------|------|
| **Complexity** | Simple | Moderate |
| **Extension factor** | 2x-4x | 4x-16x+ |
| **Local context preservation** | Poor | Good |
| **Fine-tuning required** | Yes (more) | Yes (less) |
| **Implementation** | Scale positions | Scale base + temperature |

**Use Linear Scaling when:**
- You need a quick extension (2x)
- You plan to fine-tune extensively
- Simplicity is preferred

**Use YaRN when:**
- You need larger extensions (4x+)
- You want to preserve local context quality
- You have limited fine-tuning budget

## Code Walkthrough

### Step 1: Understanding the Frequency Spectrum

```python
d_model = 64
base = 10000.0
inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
print(f"Frequency range: {inv_freq[-1]:.6f} to {inv_freq[0]:.6f}")
# Output: Frequency range: 0.000100 to 1.000000
# Low index = high frequency (fast rotation)
# High index = low frequency (slow rotation)
```

### Step 2: Apply Linear Scaling

```python
rope = ScaledRoPE(d_model=64, scale_factor=2.0)
x = torch.randn(1, 100, 64)  # 100 tokens
out = rope(x)  # Positions are halved: 0, 0.5, 1.0, ..., 49.5
```

### Step 3: Apply YaRN

```python
yarn = YaRNRope(d_model=64, scale_factor=4.0)
x = torch.randn(1, 400, 64)  # 400 tokens
out = yarn(x)  # NTK-aware scaling + temperature
```

### Step 4: Comparing Outputs

```python
x = torch.randn(1, 50, 64)
rope_std = ScaledRoPE(d_model=64, scale_factor=1.0)  # Standard RoPE
rope_scaled = ScaledRoPE(d_model=64, scale_factor=2.0)
yarn = YaRNRope(d_model=64, scale_factor=4.0)

out_std = rope_std(x)
out_scaled = rope_scaled(x)
out_yarn = yarn(x)

# Different approaches produce different encodings
print(f"Standard vs Scaled: {(out_std - out_scaled).abs().mean():.4f}")
print(f"Standard vs YaRN:   {(out_std - out_yarn).abs().mean():.4f}")
```

## Practical Tips

### 1. Fine-tuning After Scaling

Context extension techniques work best with some fine-tuning:

```python
# Typical fine-tuning setup
model = load_pretrained_model("llama-2-7b")
model.rope = YaRNRope(d_model=4096, scale_factor=4.0)

# Fine-tune on long sequences
for batch in long_sequence_dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### 2. Progressive Extension

For very large extensions (e.g., 4K -> 128K), consider progressive steps:

```
4K -> 16K (4x) -> 64K (4x) -> 128K (2x)
```

Each step requires less fine-tuning than a single large jump.

### 3. Evaluation

Always evaluate on:
- **Perplexity** at various sequence lengths
- **Retrieval tasks**: Can the model find information at position 50K?
- **Generation quality**: Does long-context generation remain coherent?

### 4. Memory Considerations

Longer contexts mean more memory for the KV cache:

```
Memory = 2 * n_layers * n_heads * seq_len * d_k * sizeof(float16)
```

For a 70B model at 128K context, this can be 40+ GB.

## Common Pitfalls

1. **Scaling too aggressively**: Going from 4K to 256K without fine-tuning will produce garbage
2. **Ignoring the attention temperature**: YaRN without temperature scaling may produce overconfident attention
3. **Not evaluating properly**: Perplexity alone doesn't capture retrieval quality
4. **Forgetting about KV cache memory**: Long context needs proportional memory

## Key Takeaways

1. **Context extension is essential** for practical LLM applications
2. **Linear scaling** is simple but loses local context quality
3. **NTK-aware scaling** preserves high frequencies by adjusting the base
4. **YaRN** combines NTK scaling with attention temperature for best results
5. **Fine-tuning** is almost always needed after context extension
6. **Progressive extension** is better than large jumps

## Next Steps

- **RoPE variants**: ALiBi, Kerple, FIRE
- **Efficient attention**: Flash Attention, Ring Attention for very long sequences
- **Sparse attention**: Longformer, BigBird for mixed local/global patterns
- **Retrieval-augmented**: RETRO, Infini-attention for truly unlimited context

## References

- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Position Interpolation
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — YaRN
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Original RoPE
- [LLaMA 2 Long](https://arxiv.org/abs/2307.09288) — Scaled RoPE in practice
