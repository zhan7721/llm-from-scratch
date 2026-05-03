# Transformer Block: Pre-Norm, SwiGLU, and RMSNorm

> **Module 01 -- Foundations, Chapter 04**

A transformer model is built by stacking identical **transformer blocks**. Each block takes a sequence of token representations and refines them through two sub-layers: attention (mixing information across positions) and a feed-forward network (transforming each position independently). In this chapter we implement a modern transformer block using the design choices found in LLaMA, PaLM, and other state-of-the-art models:

1. **RMSNorm** -- a simpler, faster alternative to LayerNorm
2. **SwiGLU** -- a gated feed-forward network that outperforms ReLU FFNs
3. **Pre-Norm residual structure** -- normalization before each sub-layer for training stability

---

## Prerequisites

- Understanding of attention mechanisms (Chapter 03)
- Understanding of embeddings (Chapter 02)
- Basic PyTorch: `nn.Module`, `nn.Linear`, `nn.Parameter`

## Files

| File | Purpose |
|------|---------|
| `transformer_block.py` | Core implementation of RMSNorm, SwiGLU, and TransformerBlock |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A single transformer block transforms an input sequence as follows:

```
Input: x (batch_size, seq_len, d_model)

    x = x + Attention(RMSNorm(x))      # sub-layer 1: self-attention
    x = x + SwiGLU(RMSNorm(x))         # sub-layer 2: feed-forward

Output: x (batch_size, seq_len, d_model)
```

The `+` signs are **residual connections** -- they add the input directly to the output of each sub-layer. This is critical for training deep networks.

In a full LLM, you stack many of these blocks (e.g., 32 layers for LLaMA 7B), optionally adding RoPE positional embeddings at the input.

---

## RMSNorm: Root Mean Square Normalization

### What It Does

RMSNorm normalizes the input vector so that its "energy" (root mean square) is approximately 1, then applies a learnable scale:

```
RMSNorm(x) = x / RMS(x) * weight
where RMS(x) = sqrt(mean(x^2) + eps)
```

### Comparison with LayerNorm

LayerNorm (used in the original Transformer and GPT-2) performs two steps:
1. **Center** the input by subtracting the mean
2. **Scale** by dividing by the standard deviation

```
LayerNorm(x) = (x - mean(x)) / std(x) * weight + bias
```

RMSNorm skips step 1 (no mean subtraction) and uses RMS instead of standard deviation:

| Property | LayerNorm | RMSNorm |
|----------|-----------|---------|
| Mean subtraction | Yes | No |
| Denominator | Std deviation | Root mean square |
| Bias parameter | Yes | No |
| Learnable scale | Yes | Yes |
| Computation | More expensive | Cheaper |

### Why RMSNorm Works

The original paper (Zhang & Sennrich, 2019) showed that the re-centering (mean subtraction) in LayerNorm is not essential. What matters most is the re-scaling -- keeping the magnitude of activations under control. RMSNorm provides this at lower computational cost.

Empirically, RMSNorm performs as well as or better than LayerNorm across many tasks. It has been adopted by LLaMA, PaLM, Gemma, and most modern LLMs.

### Code Walkthrough

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))  # learnable scale
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

Key details:
- `x.pow(2).mean(-1, keepdim=True)` computes the mean of squared values along the last dimension (the feature dimension). `keepdim=True` preserves the dimension for broadcasting.
- Adding `eps` before the square root prevents division by zero.
- `self.weight` is initialized to all ones, so at the start of training, RMSNorm is approximately an identity function.

---

## SwiGLU: Gated Feed-Forward Network

### The Standard FFN

In the original Transformer, each block has a position-wise feed-forward network:

```
FFN(x) = W2 * ReLU(W1 * x + b1) + b2
```

This projects the input to a higher dimension (typically 4x), applies a nonlinearity, then projects back. The hidden dimension is usually `d_ff = 4 * d_model`.

### The Problem with ReLU

ReLU sets all negative values to zero. This means half the hidden units are "dead" for any given input. While this sparsity can be useful, it limits the network's expressiveness.

### SwiGLU: Gating Is Better

SwiGLU (Shazeer, 2020) replaces the simple ReLU activation with a **gated** mechanism:

```
SwiGLU(x) = W2 * (SiLU(W_gate * x) * W1 * x)
```

where:
- `W1 * x` is the "value" -- the information to pass through
- `W_gate * x` is the "gate" -- controls how much of the value to let through
- `SiLU` (also called Swish) is the activation: `SiLU(x) = x * sigmoid(x)`
- The `*` is element-wise multiplication (the gating operation)

### Why SiLU Instead of ReLU?

SiLU has several advantages over ReLU:
- **Smooth**: No sharp corner at zero, which helps gradient-based optimization
- **Non-monotonic**: It goes slightly negative before rising, allowing small negative outputs
- **Self-gated**: `SiLU(x) = x * sigmoid(x)` -- the input acts as its own gate

### Why the Hidden Dimension Formula

With SwiGLU, the default hidden dimension is:

```
d_ff = (2/3) * 4 * d_model    (rounded to nearest multiple of 256)
```

This is because SwiGLU has **three** weight matrices (W1, W_gate, W2) instead of two (W1, W2). Using `d_ff = 4 * d_model` would make the FFN 50% larger in parameters. The `(2/3)` factor brings the parameter count back to roughly the same as the standard FFN.

For example, with `d_model = 4096`:
- Standard FFN: `d_ff = 16384`, 2 weight matrices -> `2 * 4096 * 16384 = 134M` params
- SwiGLU FFN: `d_ff = 10922` (rounded to 11008), 3 weight matrices -> `3 * 4096 * 11008 = 135M` params

Nearly the same parameter count, but SwiGLU performs better.

### Code Walkthrough

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = ((d_ff + 255) // 256) * 256  # round to multiple of 256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))
```

The forward pass:
1. `self.w_gate(x)` produces the gate, activated by SiLU
2. `self.w1(x)` produces the value
3. Element-wise multiply: `SiLU(gate) * value`
4. Project back to `d_model` with `self.w2`

Note: Modern LLMs use `bias=False` in all linear layers. This reduces parameters and has no measurable quality impact.

---

## Pre-Norm vs Post-Norm

### Post-Norm (Original Transformer)

The original Transformer paper (2017) applies normalization **after** each sub-layer:

```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

### Pre-Norm (GPT-2, LLaMA, and beyond)

Modern LLMs apply normalization **before** each sub-layer:

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Why Pre-Norm Is Better

#### 1. Gradient Flow

In Post-Norm, the gradient must pass through the LayerNorm at every layer. For a network with L layers, the gradient is multiplied by L normalization operations, each of which can shrink or distort it. This makes training unstable for deep models.

In Pre-Norm, the residual connection provides a **direct gradient path** from the output to the input, bypassing the sub-layers entirely. The gradient can flow unimpeded from the last layer to the first:

```
d_output / d_input = 1 + d_sublayer / d_input
                     ^            ^
                     |            |
                direct path   contribution from sub-layer
```

The `1` ensures the gradient is at least as large as the upstream gradient.

#### 2. Training Stability

Post-Norm models are sensitive to:
- **Learning rate**: Too high causes divergence, too low wastes compute
- **Warmup**: A learning rate warmup schedule is essential
- **Initialization**: Poor initialization can make training fail entirely

Pre-Norm models are much more forgiving. They can train with:
- Higher learning rates
- Less warmup (or none)
- Standard initialization

#### 3. The Trade-off

Pre-Norm has one subtle disadvantage: the residual stream can "dominate" the sub-layer contributions. In very deep networks, the sub-layers may contribute relatively little to the output, as the residual connections carry most of the signal. This is sometimes called the "representation collapse" problem.

However, in practice, this is manageable and the training stability benefits far outweigh this concern. All major LLMs (LLaMA, PaLM, GPT-3, Gemma) use Pre-Norm.

### Code: The Pre-Norm Pattern

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, causal=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))   # Pre-Norm attention
        x = x + self.ffn(self.ffn_norm(x))      # Pre-Norm FFN
        return x
```

Each line follows the pattern:
1. **Normalize** the input: `self.attn_norm(x)`
2. **Transform** through the sub-layer: `self.attn(...)`
3. **Add** the residual: `x + ...`

---

## Residual Connections

### Why They Matter

Residual connections (He et al., 2015) are the single most important architectural innovation for training deep networks. They work by adding the input directly to the output:

```
output = x + F(x)
```

instead of just:

```
output = F(x)
```

### Three Key Benefits

#### 1. Gradient Highway

During backpropagation, the gradient of the loss with respect to the input includes a term that is simply `1` (the identity). This means the gradient can flow directly from the output to the input without being attenuated by the sub-layer.

Without residual connections, gradients must pass through every layer's weights and activations. In a 32-layer network, this means 32 matrix multiplications, each of which can shrink (or explode) the gradient. With residual connections, the gradient has a shortcut.

#### 2. Easier Optimization

Instead of learning a function `F(x)` that maps input to output, the sub-layer only needs to learn the **residual** `F(x) = output - x` -- the *difference* between the desired output and the input. It is easier to learn small corrections than to learn the entire transformation from scratch.

#### 3. Identity Initialization

At the start of training, with zero-initialized weights (or near-zero), `F(x) = 0`, so `output = x`. The network starts as an identity function and gradually learns to make meaningful transformations. This is a much better starting point than random initialization.

### In the Transformer Block

Each transformer block has two residual connections:
1. Around the attention sub-layer
2. Around the FFN sub-layer

This means the gradient has two shortcuts per block. For a 32-layer model, the gradient has 64 direct paths from output to input.

---

## Putting It All Together

### The Full TransformerBlock

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, causal=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)       # normalize before attention
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ffn_norm = RMSNorm(d_model)         # normalize before FFN
        self.ffn = SwiGLU(d_model, d_ff)         # gated FFN

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))    # attention + residual
        x = x + self.ffn(self.ffn_norm(x))      # FFN + residual
        return x
```

### Data Flow

For a single token at position `t`:

```
1. Input: x_t (a vector of d_model numbers)

2. Attention sub-layer:
   a. Normalize: h = RMSNorm(x_t)
   b. Attention: The model computes how much position t should attend
      to every other position (including itself), using Q/K/V projections
   c. Residual: x_t = x_t + attention_output

3. FFN sub-layer:
   a. Normalize: h = RMSNorm(x_t)
   b. Gate: g = SiLU(W_gate @ h)
   c. Value: v = W1 @ h
   d. Combine: x_t = W2 @ (g * v)
   e. Residual: x_t = x_t + ffn_output

4. Output: x_t (refined representation)
```

### Parameter Count per Block

For a block with `d_model` and `n_heads`:

| Component | Parameters |
|-----------|-----------|
| RMSNorm (x2) | 2 * d_model |
| Attention (Q, K, V, O) | 4 * d_model^2 |
| SwiGLU (W1, W_gate, W2) | 3 * d_model * d_ff |
| **Total** | ~4 * d_model^2 + 3 * d_model * d_ff + 2 * d_model |

For LLaMA 7B (d_model=4096, d_ff=11008, n_heads=32):
- Attention: 4 * 4096^2 = 67M
- SwiGLU: 3 * 4096 * 11008 = 135M
- Total per block: ~202M

---

## Design Choices in Modern LLMs

| Component | Original Transformer | LLaMA / Modern LLMs |
|-----------|---------------------|---------------------|
| Normalization | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) |
| FFN activation | ReLU | SwiGLU |
| FFN hidden dim | 4 * d_model | (2/3) * 4 * d_model |
| Linear bias | Yes | No |
| Position encoding | Absolute sinusoidal | RoPE |

Each of these changes was motivated by either improved training stability, better quality, or reduced computational cost.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/04_transformer_block/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 01-foundations/04_transformer_block/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the transformer block yourself.

### Exercise Order

1. **`RMSNorm.forward`** -- Implement RMS normalization: compute RMS, divide, scale by weight
2. **`SwiGLU.forward`** -- Implement the gated FFN: SiLU gate times linear value, then project
3. **`TransformerBlock.forward`** -- Wire up the Pre-Norm residual structure

### Tips

- Start with RMSNorm. The formula is simple: `x / sqrt(mean(x^2) + eps) * weight`. Use `keepdim=True` when computing the mean so it broadcasts correctly.
- For SwiGLU, the key insight is that there are two parallel paths: the gate (`W_gate`) and the value (`W1`). They are combined with element-wise multiplication after applying SiLU to the gate.
- For TransformerBlock, each sub-layer follows the same pattern: normalize, transform, add residual. Two lines of code.

---

## Key Takeaways

1. **RMSNorm is simpler and equally effective.** By dropping mean subtraction, it saves computation without sacrificing quality. Adopted by all major LLMs.

2. **SwiGLU replaces ReLU with a gated mechanism.** The SiLU activation on a gate controls how much of a linear projection passes through. This is more expressive than ReLU and has become the standard FFN in modern LLMs.

3. **Pre-Norm stabilizes training.** By normalizing before each sub-layer (not after), gradients flow more smoothly through the residual connections. This enables training of deeper models with higher learning rates.

4. **Residual connections are essential.** They provide a gradient highway, make optimization easier, and allow the network to start as an identity function. Every modern deep network uses them.

5. **The transformer block is the fundamental building block.** A full LLM is just a stack of these blocks, with embeddings at the input and a language model head at the output.

---

## Further Reading

- [LLaMA Paper (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) -- Uses RMSNorm, SwiGLU, Pre-Norm, RoPE
- [PaLM Paper (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) -- Uses SwiGLU at scale
- [RMSNorm Paper (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) -- Root Mean Square Layer Normalization
- [GLU Variants Paper (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) -- SwiGLU and other gated FFN variants
- [On Layer Normalization in the Transformer (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745) -- Analysis of Pre-Norm vs Post-Norm
- [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385) -- Original residual connection paper
