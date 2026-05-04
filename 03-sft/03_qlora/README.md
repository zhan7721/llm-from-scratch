# QLoRA: Quantized Low-Rank Adaptation

## Motivation

While LoRA significantly reduces the number of trainable parameters, it still requires loading the full model in FP16 precision. For a 65B parameter model, this requires ~130GB of GPU memory just for the weights, making it impossible to fine-tune on consumer hardware.

**QLoRA** (Dettmers et al., 2023) solves this by combining:
1. **4-bit NormalFloat (NF4) quantization** for base weights
2. **LoRA adapters** in full precision for training
3. **Double quantization** to further compress quantization constants
4. **Paged optimizers** to handle memory spikes

This enables fine-tuning a 65B model on a single 48GB GPU while maintaining full 16-bit fine-tuning performance.

## NF4 Quantization

### Why NF4?

Standard quantization methods (INT4, INT8) assume uniform distribution of values. However, neural network weights are approximately **normally distributed**. NF4 is information-theoretically optimal for normal distributions.

### Quantization Levels

The 16 NF4 levels are precomputed to minimize expected quantization error for standard normal data:

```
NF4_LEVELS = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
              0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

These levels are spaced to have equal probability mass under the standard normal distribution, ensuring each quantization bin contains approximately the same number of values.

### Block-wise Quantization

To handle outliers and varying scales across the tensor, we use **block-wise quantization**:

1. Divide the tensor into blocks of size B (default: 64)
2. For each block, compute the absolute maximum value as the scaling factor
3. Normalize the block to [-1, 1]
4. Quantize each normalized value to the nearest NF4 level
5. Store the 4-bit indices and per-block scales

```
Original: [0.5, -0.3, 0.8, -0.1, ...] (float32)
Scale:    0.8 (absmax of block)
Normalized: [0.625, -0.375, 1.0, -0.125, ...]
NF4 Indices: [12, 3, 15, 5, ...] (4-bit each)
```

### Memory Savings

- **FP32**: 32 bits per weight
- **FP16**: 16 bits per weight
- **NF4**: 4 bits per weight + ~0.5 bits for scales (with block_size=64)

For a 7B parameter model:
- FP32: 28 GB
- FP16: 14 GB
- NF4: ~3.5 GB + 0.05 GB scales ≈ 3.6 GB

## QLoRA Architecture

QLoRA combines NF4 quantization with LoRA:

```
                    Input x
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │ NF4 Weights │         │  LoRA Path  │
    │  (frozen)   │         │ (trainable) │
    └─────────────┘         └─────────────┘
           │                       │
           ▼                       ▼
    Dequantize to FP16      x → Dropout → A → B
           │                       │
           ▼                       ▼
      W_q @ x                B @ A @ x × (α/r)
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
                  Output + LoRA
```

### Key Properties

1. **Base weights**: Stored in NF4, frozen during training
2. **LoRA matrices**: Stored in FP16/BF16, trainable
3. **Forward pass**: Dequantize base weights on-the-fly, add LoRA contribution
4. **Backward pass**: Gradients only flow through LoRA matrices

### Memory Breakdown

For a 7B model with LoRA rank 64:
- Base weights (NF4): ~3.6 GB
- LoRA matrices (FP16): ~0.1 GB
- Optimizer states (Adam): ~0.2 GB
- Activations: ~2-4 GB
- **Total**: ~6-8 GB (fits on single GPU)

## Double Quantization

QLoRA introduces **double quantization** to further compress the quantization constants:

1. **First quantization**: Weights → NF4 with block_size=64
2. **Second quantization**: Quantize the scales themselves to FP8 with block_size=256

This reduces the memory overhead of scales from 0.5 bits/weight to ~0.127 bits/weight.

```
Scales (FP32):  0.5 bits per weight
Scales (FP8):   0.127 bits per weight
Savings:        ~0.37 bits per weight
```

## Paged Optimizers

During training, optimizer states can cause memory spikes. QLoRA uses **paged optimizers** that automatically move optimizer states to CPU memory when GPU memory is insufficient, and page them back when needed.

This is implemented using NVIDIA's unified memory feature, which provides a single address space spanning both CPU and GPU memory.

## Code Walkthrough

### NF4Quantizer

```python
class NF4Quantizer:
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
    ])

    def quantize(self, tensor):
        # 1. Reshape into blocks
        # 2. Compute per-block absmax scales
        # 3. Normalize to [-1, 1]
        # 4. Find nearest NF4 level
        # 5. Pack two 4-bit indices into one int8
        return packed, scales

    def dequantize(self, packed, scales, original_shape):
        # 1. Unpack 4-bit indices
        # 2. Look up NF4 levels
        # 3. Multiply by scales
        # 4. Reshape to original
        return dequantized_tensor
```

### QLoRALinear

```python
class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        self.quant_linear = QuantizedLinear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_output = self.quant_linear(x)  # Dequantized on-the-fly
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output
```

### apply_qlora

```python
def apply_qlora(model, rank=8, alpha=16.0, target_modules=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target in name:
            # 1. Create QLoRALinear
            # 2. Copy weights
            # 3. Quantize base weights
            # 4. Replace module
```

## Memory Savings Comparison

| Model | FP32 | FP16 | NF4 | NF4 + LoRA (r=64) |
|-------|------|------|-----|-------------------|
| 7B | 28 GB | 14 GB | 3.6 GB | ~6 GB |
| 13B | 52 GB | 26 GB | 6.5 GB | ~10 GB |
| 33B | 132 GB | 66 GB | 16.5 GB | ~22 GB |
| 65B | 260 GB | 130 GB | 32.5 GB | ~42 GB |

## References

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Large Language Models. arXiv:2305.14314.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
