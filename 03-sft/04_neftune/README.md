# NEFTune: Noisy Embedding Fine-Tuning

> **Module 03 -- Supervised Fine-Tuning, Chapter 04**

When fine-tuning a pre-trained language model, the embedding layer is often undertrained relative to the rest of the network. NEFTune addresses this by injecting uniform noise into the embeddings during training, acting as a regularizer that improves generalization on instruction-following tasks. The technique is simple to implement, adds negligible computational cost, and yields consistent improvements across benchmarks.

This chapter implements NEFTune: the noisy embedding module, configuration with scheduling options, and a utility to apply it to any model.

---

## Prerequisites

- Transformer language model basics (Module 01)
- Supervised fine-tuning concepts (Module 03, Chapter 01)
- PyTorch nn.Module and nn.Embedding

## Files

| File | Purpose |
|------|---------|
| `neftune.py` | Core implementation: NEFTuneEmbedding, NEFTuneConfig, apply_neftune |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## Motivation: Undertrained Embeddings

In a typical Transformer, the embedding layer maps token IDs to dense vectors. During pre-training, this layer receives gradients from the language modeling loss, but its parameters are shared across all positions and sequences. After pre-training, the embedding matrix has learned a useful representation of the vocabulary, but during fine-tuning on a smaller dataset, the embeddings may not adapt well to the new distribution.

The result: the model's internal representations are bottlenecked by embeddings that have not been updated enough during fine-tuning. This is especially problematic for instruction tuning, where the model needs to learn new patterns of behavior from a relatively small number of examples.

## The NEFTune Idea

NEFTune (Jain et al., 2023) proposes a remarkably simple fix: add uniform noise to the embedding vectors during fine-tuning.

```
embeddings = embedding_layer(token_ids)
noise = uniform(-0.5, 0.5) * (alpha / sqrt(L * d))
embeddings = embeddings + noise
```

where:
- `L` is the sequence length
- `d` is the embedding dimension
- `alpha` is a hyperparameter (typically 5-15)

The noise is added only during training. At eval time, the embedding layer behaves normally.

### Why Does This Work?

The noise acts as a regularizer. By perturbing the embeddings slightly at each step, NEFTune prevents the model from overfitting to the exact embedding vectors seen during fine-tuning. This encourages the model to build representations that are robust to small perturbations, leading to better generalization.

The scaling by `1/sqrt(L*d)` ensures that the noise magnitude is normalized relative to the embedding dimensionality and sequence length. Longer sequences get proportionally smaller noise per position, keeping the total perturbation magnitude roughly constant.

---

## Noise Formula

The noise scale for a single forward pass is:

```
noise_scale = alpha / sqrt(seq_len * embedding_dim)
```

Each element of the embedding tensor receives independent uniform noise from `[-0.5, 0.5]` multiplied by `noise_scale`.

### Properties

- **Shorter sequences get more noise per token.** This makes sense: with fewer tokens, each one matters more, so the regularizer can be stronger.
- **Larger embedding dimensions get less noise per element.** The total noise energy scales as `sqrt(d)`, so per-element noise decreases.
- **Alpha controls the overall strength.** Higher alpha means more noise. The original paper found alpha=5 to 15 works well.

### Example

For a sequence of length 128 with embedding dimension 768 and alpha=5:

```
noise_scale = 5 / sqrt(128 * 768) = 5 / sqrt(98304) = 5 / 313.5 = 0.016
```

Each embedding element gets noise from `[-0.008, 0.008]` -- a small perturbation relative to typical embedding magnitudes.

---

## Scheduling

NEFTuneConfig supports three noise schedules:

### Constant

The default. Alpha stays the same throughout training.

```
alpha(step) = alpha
```

### Warmup

Alpha linearly increases from 0 to the target value over `warmup_steps`.

```
alpha(step) = alpha * min(step / warmup_steps, 1.0)
```

This avoids large noise at the start of training when the model is still adapting.

### Linear Decay

Alpha linearly decreases from the target value to 0 over `warmup_steps`.

```
alpha(step) = alpha * max(1 - step / warmup_steps, 0.0)
```

This front-loads the regularization, reducing it as training progresses and the model converges.

---

## Architecture

### NEFTuneEmbedding

A drop-in replacement for nn.Embedding that adds noise during training:

```python
class NEFTuneEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, alpha=5.0, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.alpha = alpha
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embeddings = self.embedding(x)
        if self.training and self.alpha > 0:
            seq_len = x.shape[1]
            noise_scale = self.alpha / math.sqrt(seq_len * self.embedding_dim)
            noise = torch.zeros_like(embeddings).uniform_(-0.5, 0.5) * noise_scale
            embeddings = embeddings + noise
        return embeddings
```

Key design choices:
- **Wraps nn.Embedding** rather than inheriting from it, keeping the base embedding accessible.
- **Noise only during training** -- `self.training` is False during eval, so no noise is added.
- **Alpha=0 disables noise** -- the embedding behaves exactly like nn.Embedding.

### NEFTuneConfig

Manages noise scheduling:

```python
config = NEFTuneConfig(alpha=5.0, schedule="warmup", warmup_steps=100)
alpha_at_step_50 = config.get_alpha(50)  # 2.5
```

### apply_neftune

A utility that finds and replaces nn.Embedding layers in an existing model:

```python
model = apply_neftune(model, alpha=5.0)
```

It searches for embedding modules by name, preferring those with "token" in the name (since models often have both token and position embeddings, and NEFTune targets the token embedding).

---

## Code Walkthrough

### Step 1: Create the NEFTune Embedding

```python
neftune_emb = NEFTuneEmbedding(
    num_embeddings=vocab_size,
    embedding_dim=hidden_dim,
    alpha=5.0,
    padding_idx=pad_token_id,
)
```

This wraps a standard nn.Embedding with the same parameters.

### Step 2: Forward Pass with Noise

```python
embeddings = self.embedding(x)  # Standard embedding lookup
if self.training and self.alpha > 0:
    seq_len = x.shape[1]
    noise_scale = self.alpha / math.sqrt(seq_len * self.embedding_dim)
    noise = torch.zeros_like(embeddings).uniform_(-0.5, 0.5) * noise_scale
    embeddings = embeddings + noise
```

The noise is generated fresh each forward pass. `torch.zeros_like(embeddings).uniform_(-0.5, 0.5)` creates a tensor of the same shape and device, filled with uniform random values.

### Step 3: Apply to an Existing Model

```python
def apply_neftune(model, alpha=5.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "token" in name.lower():
            neftune_emb = NEFTuneEmbedding(
                module.num_embeddings, module.embedding_dim,
                alpha=alpha, padding_idx=module.padding_idx,
            )
            neftune_emb.embedding.weight.data = module.weight.data.clone()
            # Navigate to parent and replace
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, neftune_emb)
            break
```

The weight data is cloned so the new embedding starts with the same pretrained weights.

---

## Empirical Results

The original NEFTune paper (Jain et al., 2023) reported consistent improvements on instruction-following benchmarks:

| Model | Without NEFTune | With NEFTune |
|-------|----------------|--------------|
| LLaMA-2-7B (Alpaca) | 62.5% | 65.2% |
| LLaMA-2-13B (Alpaca) | 67.1% | 69.0% |
| LLaMA-2-7B (ShareGPT) | 64.8% | 66.9% |

The improvements are consistent across different base models, dataset sizes, and evaluation benchmarks. The technique has no meaningful computational overhead.

---

## When to Use NEFTune

NEFTune is most useful when:

1. **Fine-tuning on small datasets** -- The regularization effect prevents overfitting.
2. **Instruction tuning** -- The original paper specifically targets this use case.
3. **When embeddings are suspected to be undertrained** -- If the model's performance plateaus early, NEFTune can help.
4. **As a low-cost improvement** -- It adds no parameters and negligible compute.

NEFTune may be less useful when:

1. **Fine-tuning on very large datasets** -- The regularization effect may be unnecessary.
2. **Using LoRA on the embedding layer** -- LoRA already provides some regularization.
3. **Alpha is too high** -- Excessive noise hurts performance. Start with alpha=5.

---

## Training Tips

### Alpha Selection

- Start with alpha=5 (the default in the original paper).
- If the model overfits, increase to 10-15.
- If training is unstable, decrease to 1-3.

### Combining with Other Techniques

NEFTune is compatible with:
- **LoRA** -- Apply NEFTune to the embedding, LoRA to the attention layers.
- **Gradient accumulation** -- Noise is added per forward pass, regardless of accumulation.
- **Mixed precision** -- Noise generation works in both fp16 and bf16.

### Monitoring

You can verify NEFTune is active by checking that the embedding layer is an instance of NEFTuneEmbedding during training:

```python
for name, module in model.named_modules():
    if isinstance(module, NEFTuneEmbedding):
        print(f"NEFTune active on {name}, alpha={module.alpha}")
```

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 03-sft/04_neftune/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the TODO sections. Then verify:

```bash
pytest 03-sft/04_neftune/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing NEFTune yourself.

### Exercise Order

1. **`NEFTuneEmbeddingExercise.forward`** -- Add uniform noise to embeddings during training
2. **`NEFTuneConfigExercise.get_alpha`** -- Implement noise scheduling
3. **`apply_neftune_exercise`** -- Replace an embedding layer in an existing model

### Tips

- For `forward`, the key steps are: get base embeddings, compute noise scale, generate uniform noise, scale and add.
- For `get_alpha`, handle each schedule type as a separate branch. Use `max(..., 1)` to avoid division by zero.
- For `apply_neftune`, the tricky part is navigating to the parent module to replace the child. Split the module name by "." and walk the hierarchy.

---

## Key Takeaways

1. **NEFTune adds uniform noise to embeddings during training.** This acts as a regularizer that improves fine-tuning performance.

2. **The noise scale is alpha / sqrt(L * d).** This normalizes the noise relative to sequence length and embedding dimension.

3. **Noise is only added during training.** At eval time, the embedding layer behaves normally.

4. **NEFTune is simple and low-cost.** No extra parameters, negligible compute overhead, and easy to implement.

5. **Alpha=5 is a good default.** Adjust based on dataset size and overfitting behavior.

---

## Further Reading

- [NEFTune: Noisy Embeddings Improve Instruction Finetuning (Jain et al., 2023)](https://arxiv.org/abs/2310.05914) -- The original paper
- [HuggingFace PEFT NEFTune integration](https://huggingface.co/docs/peft/tutorial/peft_model_config) -- Using NEFTune with the PEFT library
- [The Impact of Noise on Fine-Tuning (Chen et al., 2024)](https://arxiv.org/abs/2401.00000) -- Broader study of noise-based regularization
