# Low-Rank Adaptation (LoRA)

## Motivation

Full fine-tuning of a large language model updates every parameter in the network. For a 7B-parameter model, this means storing and computing gradients for 7 billion floats -- requiring tens of gigabytes of GPU memory and significant compute.

But here is the key insight: **most of those parameters are redundant for any given task.** Research by Aghajanyan et al. (2020) showed that pretrained models have a very low "intrinsic dimensionality" -- the number of truly independent directions needed to solve a downstream task is much smaller than the parameter count suggests.

LoRA exploits this by learning a **low-rank update** to the weight matrices instead of the full update.

## The LoRA Idea

Instead of updating a weight matrix W of shape (d_out, d_in), LoRA decomposes the update into two small matrices:

```
Delta_W = B @ A
```

where:
- A has shape (rank, d_in)
- B has shape (d_out, rank)
- rank << min(d_out, d_in)

The forward pass becomes:

```
y = x @ W^T + (alpha / rank) * x @ A^T @ B^T
```

The original W is **frozen** -- only A and B receive gradients and are updated during training.

## Math

Given a pretrained weight matrix W_0, the adapted weight is:

```
W' = W_0 + (alpha / rank) * B @ A
```

At initialization:
- A is initialized with Kaiming uniform (random)
- B is initialized with **zeros**

This means at the start of training, Delta_W = 0, so the model output is identical to the pretrained model. Training then gradually learns the adaptation.

### Scaling Factor

The scaling factor `alpha / rank` controls how large the LoRA update is relative to the original weights. A common setting is alpha = 2 * rank, giving a scaling of 2.0.

- **Higher alpha** = larger LoRA update, more expressive but less stable
- **Lower alpha** = smaller LoRA update, more conservative

## Why It Works

### Intrinsic Dimensionality

Pretrained models have learned representations that are already close to optimal for many tasks. The "distance" in weight space from the pretrained model to a good fine-tuned model often lies in a low-dimensional subspace.

LoRA restricts the weight update to this low-dimensional subspace by construction. With rank=8, each weight update can only move in 8 independent directions -- but these are often enough.

### Parameter Efficiency

For a weight matrix of size (d_out, d_in):
- Full fine-tuning: d_out * d_in parameters
- LoRA with rank r: r * (d_in + d_out) parameters

For a 4096x4096 weight matrix with rank=8:
- Full: 16.7M parameters
- LoRA: 65.5K parameters (256x reduction)

## Hyperparameters

### Rank (r)

The rank of the low-rank decomposition. Common values: 4, 8, 16, 32, 64.

- Lower rank = fewer parameters, faster training, less expressive
- Higher rank = more parameters, slower training, more expressive
- For most tasks, rank 8-16 is sufficient

### Alpha

Controls the magnitude of the LoRA update. The effective update is scaled by `alpha / rank`.

- Common setting: alpha = 2 * rank (so scaling = 2.0)
- Some practitioners use alpha = rank (scaling = 1.0)

### Target Modules

Which layers to apply LoRA to. Common choices:

- **Attention only**: W_q, W_k, W_v, W_o (most common in literature)
- **Attention + FFN**: Also include w1, w2, w_gate in the feed-forward network
- **All linear layers**: Most comprehensive

The original LoRA paper found that adapting both W_q and W_v gave good results, but modern practice often targets all attention projections.

### Dropout

LoRA-specific dropout applied to the input before the LoRA path. Helps prevent overfitting, especially with small datasets. Typical values: 0.0 to 0.1.

## Merge and Deployment

One of LoRA's biggest advantages: after training, you can **merge** the LoRA weights back into the original model:

```
W_final = W_0 + (alpha / rank) * B @ A
```

After merging:
- The model has the same architecture as the original
- No extra parameters or computation at inference time
- The merged model can be used as a drop-in replacement

This means **zero inference overhead** -- the adapted model runs at exactly the same speed as the base model.

You can also keep multiple LoRA adapters for different tasks and swap them at runtime, or merge different LoRAs to combine capabilities.

## Code Walkthrough

### LoRALinear

The core building block. Wraps a frozen `nn.Linear` with trainable A and B matrices:

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0, bias=True):
        # Frozen original layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # B starts at zero, so initial output matches the frozen model
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
```

Forward pass adds the LoRA contribution:

```python
def forward(self, x):
    orig_output = self.linear(x)
    lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
    return orig_output + lora_output
```

### apply_lora

Replaces target `nn.Linear` modules with `LoRALinear`:

```python
def apply_lora(model, rank=8, alpha=16.0, target_modules=None, dropout=0.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # Replace with LoRA version, preserving original weights
            ...
```

### Merge/Unmerge

Merging folds LoRA into the base weights for zero-overhead inference:

```python
def merge(self):
    self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling

def unmerge(self):
    self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
```

## Comparison: LoRA vs Full Fine-Tuning vs Prefix Tuning

| Aspect | Full Fine-Tuning | LoRA | Prefix Tuning |
|--------|-----------------|------|---------------|
| Trainable params | 100% | 0.1-1% | 0.1-1% |
| GPU memory | High | Medium | Low |
| Training speed | Slow | Fast | Fast |
| Inference overhead | None | None (after merge) | Slight |
| Expressiveness | Full | Good | Limited |
| Multi-task | Need copies | Swap adapters | Swap prefixes |
| Implementation | Simple | Moderate | Moderate |

LoRA hits a sweet spot: it is much more memory-efficient than full fine-tuning while preserving most of the expressiveness. Unlike prefix tuning, it modifies the model weights directly and can be merged for zero-overhead inference.

## References

- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
- Aghajanyan et al. (2020). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." arXiv:2012.13255
