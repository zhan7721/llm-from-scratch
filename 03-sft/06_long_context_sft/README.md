# Long Context SFT

> **Module 03 -- Supervised Fine-Tuning, Chapter 06**

Most language models are pretrained with a fixed context window (e.g., 2048 or 4096 tokens). At inference time, they cannot attend to information beyond that window. Long-context supervised fine-tuning extends the effective context length by fine-tuning on longer sequences, enabling the model to process documents, conversations, and codebases that exceed the original limit.

This chapter implements the core techniques: Position Interpolation, NTK-aware scaling, long-context data handling with sliding windows, and chunked loss computation.

---

## Prerequisites

- Transformer language model basics (Module 01)
- Pre-training loop and loss computation (Module 02)
- Attention mechanisms and positional encoding
- PyTorch Dataset and DataLoader

## Files

| File | Purpose |
|------|---------|
| `long_context_sft.py` | Core implementation: PositionInterpolation, NTKAwareScaling, LongContextDataset, compute_long_context_loss, LongContextTrainer |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Problem: Fixed Context Windows

A transformer model trained on sequences of length L cannot generalize to sequences longer than L at inference time. The positional encodings -- whether learned, sinusoidal, or rotary (RoPE) -- have only been seen for positions 0 through L-1. When the model encounters position L, L+1, etc., the embeddings are out-of-distribution, and the model produces unreliable outputs.

This is a fundamental limitation. A model trained on 2048-token sequences cannot read a 10,000-token document. The context window is a hard boundary.

### Why Not Just Train on Longer Sequences?

You can, but it is expensive:

1. **Quadratic attention cost**: Self-attention is O(n^2) in sequence length. Doubling the context quadruples the compute.
2. **Memory**: Long sequences require proportionally more GPU memory for activations, KV cache, and gradients.
3. **Data**: You need enough long documents to fill the training batches.

Long-context SFT offers a middle ground: take a model already pretrained at a shorter length, and extend it to handle longer sequences with relatively little additional training.

---

## Position Interpolation (PI)

### The Idea

Position Interpolation (Chen et al., 2023) is the simplest approach. If a model was pretrained with a maximum position of L_orig and you want to extend it to L_target, you linearly scale all position indices by the factor L_orig / L_target.

```
Original positions:  0, 1, 2, ..., 2047       (range [0, 2048))
Interpolated:        0, 0.25, 0.5, ..., 2047.75  (range [0, 2048))
```

The model sees the same positional range it was trained on, but the positions are compressed. Information that was at position 4096 now maps to position 1024 -- the model can attend to it because it is within the original range.

### Why It Works

The key insight is that the model's positional embeddings are continuous. RoPE, for example, applies a rotation matrix based on the position index. If you feed it position 0.5, it gives a rotation halfway between position 0 and position 1 -- which the model has seen during pretraining. The interpolated positions are always within the training distribution.

### Trade-offs

- **Simple**: Just divide positions by a scale factor.
- **Effective**: Can extend context by 2-4x with minimal fine-tuning.
- **Lossy**: Compressing positions means the model loses some ability to distinguish nearby tokens. Two positions that were 1 apart are now 0.25 apart.

---

## NTK-Aware Scaling

### The Idea

NTK-aware scaling (bloc97, 2023) takes a different approach. Instead of scaling the positions, it scales the base frequency of RoPE.

RoPE uses a set of sinusoidal functions with different frequencies. Low frequencies capture local relationships (nearby tokens), while high frequencies capture global structure (distant tokens). Position Interpolation compresses all frequencies equally, which hurts the high-frequency components.

NTK-aware scaling adjusts the base frequency so that the low-frequency components are scaled (extending the context window) while the high-frequency components are preserved (maintaining local discrimination).

### The Math

Standard RoPE computes inverse frequencies as:

```
inv_freq[i] = 1.0 / (base^(2i/d))
```

NTK-aware scaling replaces the base:

```
scaled_base = base * scale_factor
inv_freq[i] = 1.0 / (scaled_base^(2i/d))
```

The higher base means lower frequencies are more compressed (extending context), while higher frequencies are less affected (preserving local structure).

### YaRN: Combining NTK with Attention Temperature

YaRN (Yet another RoPE extensioN) extends NTK-aware scaling by also adjusting the attention temperature. The softmax in attention is divided by sqrt(d), and YaRN scales this by a temperature factor that depends on the sequence length. This helps the model calibrate its attention distribution over longer sequences.

---

## Long-Context Data Handling

### Document Packing

Short documents can be packed together to fill a sequence:

```
[doc1_tokens] [SEP] [doc2_tokens] [SEP] [padding]
```

This maximizes GPU utilization by avoiding wasted padding.

### Sliding Window

Long documents that exceed the context window are processed with a sliding window:

```
Document: [t0, t1, t2, ..., t1000]
Window 1: [t0, t1, ..., t511]
Window 2: [t256, t257, ..., t767]
Window 3: [t512, t513, ..., t1000]
```

The stride controls the overlap between windows. More overlap means the model sees each token in more contexts, but requires more training steps.

### Chunked Loss Computation

For very long sequences that do not fit in GPU memory for a single forward pass, the loss can be computed in chunks:

1. Split the sequence into chunks of size `chunk_size`.
2. Compute the loss for each chunk independently.
3. Average the losses across chunks.

This is an approximation -- the model cannot attend across chunk boundaries -- but it allows training on sequences much longer than what fits in memory.

---

## Architecture

### PositionInterpolation

```python
class PositionInterpolation:
    def __init__(self, original_max_seq_len, target_max_seq_len):
        self.scale_factor = target_max_seq_len / original_max_seq_len

    def rescale_positions(self, positions):
        return positions / self.scale_factor
```

### NTKAwareScaling

```python
class NTKAwareScaling:
    def __init__(self, original_max_seq_len, target_max_seq_len, base=10000.0):
        self.scale_factor = target_max_seq_len / original_max_seq_len
        self.scaled_base = base * (self.scale_factor ** (2 * pi / (2 * pi)))

    def get_scaled_inv_freq(self, d_model, device):
        inv_freq = 1.0 / (self.scaled_base ** (arange(0, d_model, 2) / d_model))
        return inv_freq
```

### LongContextDataset

```python
class LongContextDataset(Dataset):
    def __init__(self, documents, seq_len, separator_id=0, stride=None):
        self.chunks = []
        for doc in documents:
            if len(doc) <= seq_len:
                # Pad short documents
                self.chunks.append(padded_doc[:seq_len + 1])
            else:
                # Sliding window for long documents
                for start in range(0, len(doc) - seq_len, stride):
                    self.chunks.append(doc[start:start + seq_len + 1])

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}
```

### LongContextTrainer

The trainer combines everything: position scaling, gradient accumulation, and gradient clipping.

```python
class LongContextTrainer:
    def __init__(self, model, optimizer, original_max_seq_len=2048,
                 target_max_seq_len=8192, scaling_method="pi",
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        ...
```

---

## Code Walkthrough

### Step 1: Position Interpolation

```python
pi = PositionInterpolation(original_max_seq_len=2048, target_max_seq_len=8192)
positions = torch.arange(8192)
rescaled = pi.rescale_positions(positions)
# rescaled[8191] = 8191 / 4 = 2047.75  (within original range)
```

### Step 2: NTK-Aware Scaling

```python
ntk = NTKAwareScaling(original_max_seq_len=2048, target_max_seq_len=8192)
inv_freq = ntk.get_scaled_inv_freq(d_model=64, device=torch.device("cpu"))
# inv_freq uses scaled_base instead of 10000.0
```

### Step 3: Dataset Construction

```python
documents = [list(range(100)), list(range(200))]
dataset = LongContextDataset(documents, seq_len=50)
# Short docs are padded; long docs are split with sliding window
item = dataset[0]
# item["input_ids"].shape = (50,)
# item["labels"].shape = (50,)  (shifted by 1)
```

### Step 4: Chunked Loss

```python
model = DummyModel()
batch = {"input_ids": torch.randint(0, 100, (2, 100)),
         "labels": torch.randint(0, 100, (2, 100))}
loss = compute_long_context_loss(model, batch, chunk_size=50)
# Processes the 100-token sequence in two chunks of 50
```

### Step 5: Training Loop

```python
trainer = LongContextTrainer(model, optimizer,
                             original_max_seq_len=64,
                             target_max_seq_len=128)
batch = {"input_ids": ..., "labels": ...}
result = trainer.train_step(batch, step=0)
# Returns {"loss": <scalar>}
```

---

## Training Considerations

### Gradient Accumulation

Long sequences consume more memory per batch. Gradient accumulation allows you to simulate a larger effective batch size:

```python
trainer = LongContextTrainer(
    model, optimizer,
    gradient_accumulation_steps=4,  # effective batch = 4 * micro_batch
)
```

The trainer divides the loss by the accumulation steps, accumulates gradients, and only updates weights every N steps.

### Gradient Clipping

Long sequences can produce large gradients, especially early in training when the model is adjusting to the new context length. The trainer clips gradients to `max_grad_norm` (default 1.0) to stabilize training.

### Learning Rate Schedule

Long-context fine-tuning typically uses a lower learning rate than pre-training:

- Start with a short warmup (100-500 steps)
- Use cosine decay to the final learning rate
- A typical range is 1e-5 to 5e-5

### Data Mix

Do not train exclusively on long documents. Mix in shorter documents to prevent catastrophic forgetting of short-context capabilities. A common ratio is 50% long documents, 50% short documents.

### Progressive Extension

Rather than jumping from 2K to 32K in one step, extend gradually:

```
2K -> 4K -> 8K -> 16K -> 32K
```

Each stage uses a smaller scale factor, making the adjustment easier for the model.

---

## Comparison of Methods

| Method | What Changes | Pros | Cons |
|--------|-------------|------|------|
| Position Interpolation | Scale positions linearly | Simple, effective for 2-4x | Compresses all frequencies equally |
| NTK-Aware Scaling | Scale RoPE base frequency | Preserves high frequencies | More complex, requires tuning |
| YaRN | NTK + attention temperature | Best quality | Most complex to implement |
| Direct Training | Train from scratch at long length | No approximation | Expensive, needs long data |

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 03-sft/06_long_context_sft/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 03-sft/06_long_context_sft/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing long-context SFT yourself.

### Exercise Order

1. **`PositionInterpolationExercise.rescale_positions`** -- Divide positions by the scale factor
2. **`NTKAwareScalingExercise.get_scaled_inv_freq`** -- Compute the scaled base and inverse frequencies
3. **`LongContextDatasetExercise.__getitem__`** -- Return shifted input_ids and labels from a chunk
4. **`compute_long_context_loss_exercise`** -- Compute loss in chunks for long sequences

### Tips

- Start with Position Interpolation. It is a single division operation.
- For NTK-aware scaling, the key is computing `scaled_base = base * scale_factor`.
- For `__getitem__`, remember that language modeling uses shifted labels: input_ids = chunk[:-1], labels = chunk[1:].
- For chunked loss, loop over the sequence in chunk_size steps and average the per-chunk losses.

---

## Key Takeaways

1. **Position Interpolation is the simplest extension method.** Divide all position indices by a scale factor to map them back into the original training range.

2. **NTK-aware scaling preserves high frequencies.** Instead of scaling positions, scale the RoPE base frequency. This maintains local token discrimination while extending the context window.

3. **Long documents need special handling.** Use sliding windows with overlap to process documents longer than the context window. Pack short documents together to fill batches efficiently.

4. **Chunked loss computation avoids OOM.** For sequences that do not fit in memory, compute the loss in chunks and average. This is an approximation but enables training on very long sequences.

5. **Gradient accumulation is essential.** Long sequences mean fewer examples per batch. Gradient accumulation simulates a larger effective batch size without additional memory.

6. **Extend gradually.** Going from 2K to 32K in one step is hard. Progressive extension (2K -> 4K -> 8K -> ...) is more stable and requires less fine-tuning data.

---

## Further Reading

- [Extending Context Window of Large Language Models via Position Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595) -- The PI paper
- [Reddit/bloc97 NTK-aware scaling](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) -- NTK-aware RoPE extension
- [YaRN: Efficient Context Window Extension of Large Language Models (Peng et al., 2023)](https://arxiv.org/abs/2309.00071) -- Combined NTK + temperature scaling
- [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models (Chen et al., 2023)](https://arxiv.org/abs/2309.12307) -- LoRA for long-context fine-tuning
- [Scaling Laws for RoPE (Liu et al., 2023)](https://arxiv.org/abs/2309.16739) -- Analysis of positional encoding scaling
