# Complete GPT Model Architecture

> **Module 01 -- Foundations, Chapter 05**

We have built the individual components: embeddings (Chapter 02), attention (Chapter 03), and transformer blocks (Chapter 04). Now we assemble them into a complete language model. This chapter implements a GPT-style model using LLaMA architecture choices: Pre-Norm, RMSNorm, SwiGLU, RoPE, and weight tying.

---

## Prerequisites

- Token Embedding and RoPE (Chapter 02)
- Multi-Head Attention (Chapter 03)
- Transformer Block: RMSNorm, SwiGLU, Pre-Norm (Chapter 04)
- PyTorch basics: `nn.Module`, `nn.Linear`, `dataclasses`

## Files

| File | Purpose |
|------|---------|
| `model.py` | Complete GPT model implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A GPT model is surprisingly simple at the architecture level. It takes a sequence of token IDs and produces a probability distribution over the next token at every position:

```
Input: token IDs (batch_size, seq_len)
    |
    v
TokenEmbedding          -- convert IDs to dense vectors
    |
    v
RotaryPositionalEmbedding  -- add position information
    |
    v
TransformerBlock x N    -- deep processing (attention + FFN)
    |
    v
RMSNorm                 -- final normalization
    |
    v
Linear lm_head          -- project to vocabulary
    |
    v
Output: logits (batch_size, seq_len, vocab_size)
```

The model is trained to predict the next token. Given tokens `[t1, t2, t3, t4]`, it produces logits at each position, and the loss is computed against the shifted sequence `[t2, t3, t4, t5]`.

---

## GPTConfig: Model Hyperparameters

```python
@dataclass
class GPTConfig:
    vocab_size: int = 32000     # vocabulary size
    d_model: int = 512          # hidden dimension
    n_heads: int = 8            # attention heads
    n_layers: int = 6           # transformer blocks
    d_ff: int = None            # FFN hidden dim (default: SwiGLU formula)
    max_seq_len: int = 1024     # max sequence length for RoPE
    dropout: float = 0.0        # dropout (unused, kept for compatibility)
```

### Parameter Scaling

The model size is primarily determined by `d_model` and `n_layers`. Here are some real-world configurations:

| Model | d_model | n_heads | n_layers | d_ff | Parameters |
|-------|---------|---------|----------|------|------------|
| GPT-2 Small | 768 | 12 | 12 | 3072 | 117M |
| GPT-2 Medium | 1024 | 16 | 24 | 4096 | 345M |
| LLaMA 7B | 4096 | 32 | 32 | 11008 | 6.7B |
| LLaMA 13B | 5120 | 40 | 40 | 13824 | 13B |
| LLaMA 65B | 8192 | 64 | 80 | 22016 | 65B |

Note that `n_heads` divides `d_model` evenly (each head has `d_k = d_model / n_heads` dimensions). The default `d_ff` for SwiGLU is `(2/3) * 4 * d_model`, rounded to the nearest multiple of 256.

---

## Model Architecture

### Layer 1: Token Embedding

```python
self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
```

Converts integer token IDs to dense vectors of dimension `d_model`. The output is scaled by `sqrt(d_model)` to keep the magnitude consistent regardless of the embedding dimension.

### Layer 2: Rotary Positional Embedding (RoPE)

```python
self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)
```

Encodes positional information by rotating the embedding vectors. Unlike absolute positional encodings, RoPE naturally captures relative positions through the rotation angle difference between positions.

### Layer 3: N Transformer Blocks

```python
self.layers = nn.ModuleList([
    TransformerBlock(config.d_model, config.n_heads, config.d_ff)
    for _ in range(config.n_layers)
])
```

Each block applies:
1. **Pre-Norm Attention**: `x = x + Attention(RMSNorm(x))`
2. **Pre-Norm FFN**: `x = x + SwiGLU(RMSNorm(x))`

This is where the bulk of the computation happens. Each block refines the token representations by mixing information across positions (attention) and transforming each position independently (FFN).

### Layer 4: Final RMSNorm

```python
self.norm = RMSNorm(config.d_model)
```

LLaMA applies a final RMSNorm after the last transformer block. This ensures the hidden states have appropriate magnitude before the output projection.

### Layer 5: Language Model Head

```python
self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
```

Projects the hidden states from `d_model` to `vocab_size`, producing logits for each token in the vocabulary. These logits are converted to probabilities via softmax during generation.

---

## Weight Tying

One of the most important design choices is **weight tying**:

```python
self.lm_head.weight = self.token_emb.embedding.weight
```

This makes the output projection matrix share the same parameters as the input embedding matrix. Instead of having two separate `vocab_size x d_model` matrices, we have one.

### Why Weight Tying Works

1. **Parameter reduction**: For a vocabulary of 32000 and d_model of 4096, this saves 131M parameters.

2. **Semantic consistency**: Tokens that are semantically similar should have similar embeddings AND produce similar logits. Weight tying enforces this constraint.

3. **Empirical evidence**: Press & Wolf (2017) showed that weight tying improves perplexity on language modeling benchmarks, especially for smaller models.

4. **Intuition**: The embedding matrix maps tokens to a latent space. The output projection maps from that latent space back to tokens. It makes sense for these to be inverse operations of each other.

### How It Works in PyTorch

```python
# Before tying: two separate weight matrices
assert model.lm_head.weight is not model.token_emb.embedding.weight

# After tying: same tensor object
model.lm_head.weight = model.token_emb.embedding.weight
assert model.lm_head.weight is model.token_emb.embedding.weight
```

When you call `model.parameters()`, PyTorch only returns unique parameters, so the tied weight is counted once.

---

## Forward Pass Walkthrough

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, seq_len) -- token IDs

    h = self.token_emb(x)      # (batch_size, seq_len, d_model)
    h = self.rope(h)            # (batch_size, seq_len, d_model)

    for layer in self.layers:
        h = layer(h)            # (batch_size, seq_len, d_model)

    h = self.norm(h)            # (batch_size, seq_len, d_model)
    return self.lm_head(h)      # (batch_size, seq_len, vocab_size)
```

At each position, the model produces a vector of `vocab_size` logits. Higher logits correspond to more likely next tokens.

### Shape Trace

For a batch of 2 sequences, each with 10 tokens, using d_model=64 and vocab_size=256:

```
Input:     (2, 10)           -- token IDs
Embedding: (2, 10, 64)       -- dense vectors
RoPE:      (2, 10, 64)       -- position-encoded vectors
Block 1:   (2, 10, 64)       -- refined representations
Block 2:   (2, 10, 64)       -- further refined
Norm:      (2, 10, 64)       -- normalized
lm_head:   (2, 10, 256)      -- logits for each token in vocab
```

---

## Autoregressive Generation

### The Generation Loop

Given a prompt, the model generates new tokens one at a time:

```python
@torch.no_grad()
def generate(self, prompt, max_new_tokens=100, temperature=1.0):
    for _ in range(max_new_tokens):
        # Crop to max_seq_len (handle long sequences)
        idx_cond = prompt[:, -self.config.max_seq_len:]

        # Forward pass
        logits = self(idx_cond)

        # Last position logits, with temperature
        logits = logits[:, -1, :] / temperature

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        prompt = torch.cat([prompt, next_token], dim=1)

    return prompt
```

### Temperature

Temperature controls the randomness of generation:

- **temperature = 1.0**: Standard sampling from the model's distribution.
- **temperature < 1.0** (e.g., 0.7): Makes the distribution sharper (more confident). The model is more likely to pick high-probability tokens.
- **temperature > 1.0** (e.g., 1.5): Makes the distribution flatter (more random). The model explores more diverse tokens.
- **temperature -> 0**: Approaches greedy decoding (always pick the most likely token).

Mathematically, temperature divides the logits before softmax:

```
P(token_i) = exp(logit_i / T) / sum_j(exp(logit_j / T))
```

When T < 1, the differences between logits are amplified. When T > 1, they are diminished.

### Greedy vs Sampling

**Greedy decoding** always picks the token with the highest probability. It is deterministic but tends to produce repetitive, boring text.

**Sampling** draws from the probability distribution. It produces more diverse and interesting text, but can occasionally produce nonsensical tokens.

**Top-k sampling** and **top-p (nucleus) sampling** are more advanced strategies that restrict sampling to the most likely tokens. These are not implemented here but are straightforward extensions.

### The `@torch.no_grad()` Decorator

Generation does not require gradients (we are not training). The `@torch.no_grad()` decorator tells PyTorch to skip gradient computation, which:
- Reduces memory usage (no need to store intermediate activations)
- Speeds up computation
- Is a best practice for inference

---

## Parameter Counting

### Estimating Model Size

For a model with `d_model`, `n_heads`, `n_layers`, `d_ff`, and `vocab_size`:

**Per transformer block:**
- Attention (Q, K, V, O): 4 x d_model^2
- SwiGLU (W1, W_gate, W2): 3 x d_model x d_ff
- RMSNorm (x2): 2 x d_model
- Total per block: ~4 x d_model^2 + 3 x d_model x d_ff + 2 x d_model

**Global parameters:**
- Token embedding: vocab_size x d_model
- Final RMSNorm: d_model
- lm_head: vocab_size x d_model (but tied with embedding, so 0 additional)

**Total:** n_layers x (per-block) + vocab_size x d_model + d_model

### Example: LLaMA 7B

```
d_model = 4096, n_heads = 32, n_layers = 32, d_ff = 11008, vocab_size = 32000

Per block:
  Attention: 4 x 4096^2 = 67,108,864
  SwiGLU:    3 x 4096 x 11008 = 135,266,304
  RMSNorm:   2 x 4096 = 8,192
  Total:     202,383,360

All blocks: 32 x 202,383,360 = 6,476,267,520
Embedding:  32000 x 4096 = 131,072,000
Final norm: 4096

Total: ~6,607,343,616 (approximately 6.7B parameters)
```

### Memory Estimation

In FP32 (4 bytes per parameter):
- 6.7B parameters x 4 bytes = 26.8 GB

In FP16/BF16 (2 bytes per parameter):
- 6.7B parameters x 2 bytes = 13.4 GB

During training, you also need gradients (same size as parameters) and optimizer states (2x for Adam), so the total is roughly:
- Training: ~4x parameter memory (FP32) or ~8x (mixed precision)
- Inference: ~1x parameter memory (FP16)

---

## LLaMA Design Choices Summary

Our GPT model uses the same architecture as LLaMA. Here is a summary of the key design choices:

| Component | Original Transformer (GPT-2) | LLaMA / Our Model |
|-----------|------------------------------|-------------------|
| Normalization | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) |
| FFN activation | ReLU | SwiGLU |
| FFN hidden dim | 4 x d_model | (2/3) x 4 x d_model |
| Position encoding | Absolute learned | RoPE |
| Linear bias | Yes | No |
| Weight tying | Sometimes | Yes |
| Output norm | None | RMSNorm |

Each of these changes was motivated by either improved training stability, better quality, or reduced computational cost.

### Why These Choices?

1. **RMSNorm over LayerNorm**: Cheaper computation, equally good results. Drops mean subtraction which is not essential.

2. **SwiGLU over ReLU**: Gated mechanism is more expressive. The SiLU activation allows smooth gradients and non-monotonic behavior.

3. **RoPE over absolute position**: Captures relative positions naturally. Enables length generalization with modifications (covered in Chapter 08).

4. **Pre-Norm over Post-Norm**: Better gradient flow, more stable training, enables higher learning rates.

5. **No bias in linear layers**: Reduces parameters with no measurable quality loss.

6. **Weight tying**: Reduces parameters, improves quality for smaller models.

---

## Code Walkthrough

### model.py

```python
@dataclass
class GPTConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = None
    max_seq_len: int = 1024
    dropout: float = 0.0
```

The `@dataclass` decorator automatically generates `__init__`, `__repr__`, and other methods. This is a clean way to define configuration objects.

```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.embedding.weight
```

Key implementation details:
- `nn.ModuleList` is used instead of a Python list so PyTorch properly registers the sub-modules.
- Weight tying is done by simple assignment. After this line, both `self.lm_head.weight` and `self.token_emb.embedding.weight` point to the same tensor.
- `bias=False` on the linear layer follows LLaMA convention.

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(x)
        h = self.rope(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)
```

The forward pass is a clean pipeline. Each step transforms the tensor while keeping the shape `(batch_size, seq_len, d_model)` until the final projection.

```python
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = prompt[:, -self.config.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat([prompt, next_token], dim=1)
        return prompt
```

Generation details:
- `@torch.no_grad()` disables gradient tracking for efficiency.
- `prompt[:, -self.config.max_seq_len:]` handles sequences longer than `max_seq_len` by truncating from the left.
- `logits[:, -1, :]` takes only the last position's logits (next token prediction).
- `torch.multinomial` samples from the probability distribution.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/05_model_architecture/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 01-foundations/05_model_architecture/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the GPT model yourself.

### Exercise Order

1. **`__init__`**: Create all layers (embedding, RoPE, transformer blocks, norm, lm_head)
2. **Weight tying**: Make lm_head.weight point to token_emb.embedding.weight
3. **`forward`**: Wire up the pipeline (embedding -> RoPE -> blocks -> norm -> lm_head)
4. **`generate`**: Implement the autoregressive generation loop

### Tips

- Start with `__init__`. Each component is a single line using classes from previous chapters.
- For weight tying, the key line is `self.lm_head.weight = self.token_emb.embedding.weight`. This makes them the same tensor object.
- For `forward`, think of it as a pipeline: each step takes a tensor and returns a tensor of the same shape (except the last step which projects to vocab_size).
- For `generate`, the tricky part is getting the logits for the last position: `logits[:, -1, :]`. Don't forget to apply temperature and use `torch.multinomial` for sampling.

---

## Key Takeaways

1. **GPT is simple.** The architecture is just: embedding -> N transformer blocks -> norm -> linear head. The complexity is in the components (attention, FFN), not the overall structure.

2. **Weight tying reduces parameters and improves quality.** Sharing the embedding and output projection matrices is a simple trick with significant benefits.

3. **Temperature controls generation diversity.** Lower temperature = more confident, higher temperature = more random. This is the simplest way to control generation behavior.

4. **LLaMA's design choices are now standard.** Pre-Norm, RMSNorm, SwiGLU, RoPE, no bias -- these have been validated at scale and adopted by most modern LLMs.

5. **Parameter counting is predictable.** You can estimate model size from the config alone. The dominant terms are the attention projections (4 x d_model^2 per layer) and the FFN (3 x d_model x d_ff per layer).

---

## Further Reading

- [LLaMA Paper (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) -- The architecture we implement
- [GPT-2 Paper (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -- Original GPT architecture
- [Weight Tying (Press & Wolf, 2017)](https://arxiv.org/abs/1608.05859) -- Using the same weights for input and output embeddings
- [Scaling Laws (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) -- How model size relates to performance
- [RoPE Paper (Su et al., 2021)](https://arxiv.org/abs/2104.09864) -- Rotary Positional Embedding
- [PaLM Paper (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) -- SwiGLU at scale
