# Training Loop

> **Module 02 -- Pretraining, Chapter 03**

The training loop is the engine of pretraining. It repeatedly feeds batches to the model, computes loss, updates parameters, and tracks metrics. A well-designed loop includes a learning rate schedule that warms up then decays, gradient clipping to prevent instability, gradient accumulation to simulate large batches, and mixed precision to maximize throughput. This chapter builds each component from scratch.

---

## Prerequisites

- Basic PyTorch (`nn.Module`, `torch.optim`, autograd)
- Understanding of backpropagation and gradient descent
- Familiarity with the data pipeline (Chapter 01)

## Files

| File | Purpose |
|------|---------|
| `training_loop.py` | Core implementation: scheduler, clipper, metrics, training step |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## Training Loop Components

A training step has four stages:

```
1. Forward pass:   logits = model(input_ids)
2. Loss:           loss = cross_entropy(logits, labels)
3. Backward pass:  loss.backward()
4. Parameter update:
   a. Clip gradients
   b. optimizer.step()
   c. scheduler.step()  -- adjust learning rate
   d. optimizer.zero_grad()
```

Each stage has subtleties that matter at scale.

---

## Cosine LR with Warmup

The learning rate is arguably the most important hyperparameter. Too high and training diverges; too low and training stalls. The standard schedule for LLM pretraining is **linear warmup followed by cosine decay**.

### Why warmup?

At the start of training, the model's parameters are random. Large gradients in early steps can push the model into a bad region of the loss landscape that it cannot escape. Warmup starts with a tiny learning rate and linearly increases it over the first N steps, allowing the optimizer to find a stable direction before committing to large updates.

```
LR
 ^
 |        /\
 |       /  \
 |      /    \___
 |     /         \___
 |    /              \___
 |   /                   \___
 +--+-----+-----+-----+-----> step
   0    warmup  50%    100%
```

### Cosine vs. linear decay

After warmup, the learning rate decays. Two common strategies:

- **Linear decay**: LR decreases linearly from peak to zero. Simple but can decay too aggressively early on.
- **Cosine decay**: LR follows a cosine curve, decaying slowly at first and faster later. This tends to give better final performance because the model spends more time at moderate learning rates where it can refine its representations.

The formula:

```
scale = min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(pi * progress))
```

Where `progress` goes from 0 (start of decay) to 1 (end of training). The `min_lr_ratio` (typically 0.1) prevents the LR from reaching exactly zero, which helps with fine-tuning later.

```python
class CosineLRScheduler:
    def get_lr_scale(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
```

### Practical values

- Warmup steps: typically 1-5% of total steps
- Peak LR: 3e-4 is a common default for models with 100M-1B parameters
- min_lr_ratio: 0.1 (LR decays to 10% of peak)

---

## Gradient Clipping

During training, gradients occasionally spike due to outlier batches or numerical instability. If the gradient norm becomes very large, the parameter update can be catastrophic -- the model jumps to a completely different region of the loss landscape and training collapses.

### How it works

Gradient clipping rescales the gradients when their global norm exceeds a threshold:

```
if ||g|| > max_norm:
    g = g * (max_norm / ||g||)
```

This preserves the direction of the gradient but limits its magnitude. The `max_norm` is typically 1.0.

### Norm vs. value clipping

- **Norm clipping** (what we use): Clips based on the L2 norm of all gradients combined. Preserves the relative magnitudes of different parameter gradients.
- **Value clipping**: Clips each gradient element independently. Can distort the gradient direction. Rarely used in modern training.

```python
class GradientClipper:
    def clip(self, model):
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, norm_type=self.norm_type
        )
        return total_norm.item()
```

The returned norm is useful for monitoring. If it is consistently close to `max_norm`, the model is experiencing instability and you may need to lower the learning rate.

---

## Gradient Accumulation

Large batch sizes stabilize training and improve throughput, but they require more GPU memory. Gradient accumulation simulates a larger batch by accumulating gradients over multiple forward/backward passes before updating the parameters.

```
Effective batch size = micro_batch_size * gradient_accumulation_steps

Example:
  micro_batch_size = 4
  gradient_accumulation_steps = 8
  Effective batch size = 32
```

Each micro-batch does a forward and backward pass, adding to the accumulated gradients. Only after every N micro-batches does the optimizer step. The loss must be divided by the accumulation steps so the accumulated gradients have the correct magnitude.

```python
loss = loss / gradient_accumulation_steps
loss.backward()

if (step + 1) % gradient_accumulation_steps == 0:
    clipper.clip(model)
    optimizer.step()
    optimizer.zero_grad()
```

---

## Mixed Precision Training

Modern GPUs can compute in both FP32 (32-bit float) and FP16/BF16 (16-bit float). Mixed precision uses 16-bit for the forward and backward passes (faster, less memory) while keeping a FP32 copy of the parameters for the update (more numerically stable).

### FP16 vs. BF16

- **FP16**: Has a smaller range, so gradients can underflow to zero. Requires loss scaling to prevent this.
- **BF16**: Has the same range as FP32 but less precision. Does not need loss scaling. Preferred on modern hardware (A100, H100).

### GradScaler

The `GradScaler` is needed for FP16 training. It multiplies the loss by a large factor before the backward pass (scaling up gradients to prevent underflow), then divides the gradients back before the optimizer step.

```python
scaler = torch.amp.GradScaler()

with torch.amp.autocast(device_type="cuda"):
    logits = model(input_ids)
    loss = cross_entropy(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

The `unscale_` call must happen before gradient clipping, because clipping operates on the unscaled gradients.

---

## Weight Decay

Weight decay penalizes large weights by subtracting a fraction of the weight from the gradient at each step. This acts as a regularizer, preventing the model from relying too heavily on any single feature.

### Decoupled vs. L2 regularization

- **L2 regularization**: Adds `0.5 * lambda * ||w||^2` to the loss. The gradient of this term is `lambda * w`, which is added to the gradient of the loss. This interacts with the learning rate -- changing LR also changes the effective regularization strength.
- **Decoupled weight decay** (AdamW): Subtracts `lambda * w` directly from the parameters after the optimizer step. This decouples the regularization from the learning rate, making hyperparameter tuning easier.

AdamW is the standard optimizer for LLM pretraining.

### Which parameters get weight decay?

A common practice: apply weight decay to weight matrices (dim >= 2) but not to biases and normalization parameters (dim < 2). This is because biases and norms are already constrained by their role in the network, and decaying them can hurt performance.

```python
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

optimizer = AdamW([
    {"params": decay_params, "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95))
```

---

## Code Walkthrough

### Step 1: CosineLRScheduler

```python
scheduler = CosineLRScheduler(optimizer, warmup_steps=100, total_steps=10000)
```

Creates a scheduler that linearly warms up over 100 steps, then decays with a cosine curve over the remaining 9900 steps. The `get_lr_scale` method returns a multiplier between 0 and 1 that is applied to the base learning rate.

### Step 2: GradientClipper

```python
clipper = GradientClipper(max_norm=1.0)
```

Wraps `torch.nn.utils.clip_grad_norm_` in a simple interface. The `clip` method returns the gradient norm before clipping, which is useful for logging.

### Step 3: TrainingMetrics

```python
metrics = TrainingMetrics()
metrics.update(loss=2.5, grad_norm=1.0, lr=1e-3)
print(metrics.summary())
```

Tracks loss, gradient norms, and learning rates over time. The `summary` method returns a dict with running averages and elapsed time.

### Step 4: create_training_components

```python
components = create_training_components(model, learning_rate=3e-4, total_steps=10000)
```

Creates all training components in one call. Returns a dict with `optimizer`, `scheduler`, `clipper`, and `metrics`. Handles the weight decay parameter grouping automatically.

### Step 5: training_step

```python
result = training_step(
    model, batch, optimizer, scheduler, clipper, step=0,
    gradient_accumulation_steps=1,
)
```

Executes a single training step: forward pass, loss computation, backward pass, gradient clipping, optimizer step, and LR update. Supports gradient accumulation and optional mixed precision via `GradScaler`.

---

## Putting It All Together

A typical training loop:

```python
components = create_training_components(model, total_steps=len(dataloader) * num_epochs)
optimizer = components["optimizer"]
scheduler = components["scheduler"]
clipper = components["clipper"]
metrics = components["metrics"]

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        result = training_step(model, batch, optimizer, scheduler, clipper, step)
        metrics.update(result["loss"], result["grad_norm"], result["lr"])

        if step % 100 == 0:
            print(metrics.summary())
            metrics.reset()
```

---

## Exercises

Open `exercise.py` and implement:

1. **`CosineLRScheduler.get_lr_scale`**: Implement the warmup + cosine decay formula.
2. **`GradientClipper.clip`**: Implement gradient clipping using `clip_grad_norm_`.
3. **`training_step`**: Implement the full forward/backward/step logic.

---

## Running Tests

```bash
pytest tests.py -v
```

---

## Summary

| Concept | What it solves |
|---------|---------------|
| Cosine LR + warmup | Stable start, gradual decay for better convergence |
| Gradient clipping | Prevents catastrophic updates from gradient spikes |
| Gradient accumulation | Simulates large batches with limited GPU memory |
| Mixed precision | Faster training, less memory, same quality |
| AdamW weight decay | Regularization decoupled from learning rate |
| TrainingMetrics | Monitors loss, grad norms, LR for debugging |

---

## Next Steps

- **Chapter 04 (Distributed Training)**: Scaling to multiple GPUs and machines.
- **Chapter 05 (Scaling Laws)**: How model size, data, and compute relate.
