"""QLoRA Solution: Complete reference implementation.

This file contains the complete solution for the QLoRA exercise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


# =============================================================================
# Part 1: NF4 Quantization
# =============================================================================

class NF4Quantizer:
    """NormalFloat4 (NF4) quantizer.

    NF4 is an information-theoretically optimal quantization for
    normally distributed weights. The quantization levels are precomputed
    to minimize expected quantization error for standard normal data.
    """

    # NF4 quantization levels (precomputed for standard normal)
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
    ])

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor to NF4 format.

        Steps:
        1. Reshape tensor into blocks of size self.block_size
        2. Compute per-block absmax scaling factors
        3. Normalize blocks to [-1, 1]
        4. Find nearest NF4 level for each value
        5. Pack two 4-bit indices into one int8

        Args:
            tensor: Input float tensor.

        Returns:
            (packed_indices, scales) where packed_indices contain two 4-bit values per byte.
        """
        original_shape = tensor.shape
        # Reshape to blocks
        flat = tensor.reshape(-1)
        n_blocks = (len(flat) + self.block_size - 1) // self.block_size
        padded_len = n_blocks * self.block_size
        padded = torch.zeros(padded_len, device=flat.device, dtype=flat.dtype)
        padded[:len(flat)] = flat

        blocks = padded.reshape(n_blocks, self.block_size)

        # Compute per-block scales (absmax)
        scales = blocks.abs().max(dim=1, keepdim=True).values
        scales = scales.clamp(min=1e-8)

        # Normalize blocks to [-1, 1]
        normalized = blocks / scales

        # Find nearest NF4 level for each value
        levels = self.NF4_LEVELS.to(normalized.device)
        # Expand for broadcasting: (n_blocks, block_size, 1) vs (16,)
        distances = (normalized.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0)).abs()
        indices = distances.argmin(dim=-1)  # (n_blocks, block_size)

        # Pack two 4-bit indices into one int8
        indices = indices.to(torch.uint8)
        packed = indices[:, 0::2] | (indices[:, 1::2] << 4)

        return packed, scales.squeeze(-1)

    def dequantize(self, packed: torch.Tensor, scales: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Dequantize NF4 tensor back to float.

        Args:
            packed: Packed 4-bit indices.
            scales: Per-block scales.
            original_shape: Shape of the original tensor.

        Returns:
            Dequantized float tensor.
        """
        # Unpack
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        indices = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)

        levels = self.NF4_LEVELS.to(indices.device)
        values = levels[indices.long()]

        # Rescale
        values = values * scales.unsqueeze(-1)

        # Reshape to original
        flat = values.reshape(-1)[:math.prod(original_shape)]
        return flat.reshape(original_shape)


# =============================================================================
# Part 2: QLoRA Linear Layer
# =============================================================================

class QLoRALinear(nn.Module):
    """QLoRA: Quantized linear layer with LoRA adaptation.

    The base weights are stored in NF4 format (frozen).
    LoRA A and B matrices are trained in full precision.

    Forward pass: output = NF4_dequant(W) @ x + (B @ A @ x) * scaling
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        block_size: int = 64,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # Scaling factor: alpha / rank
        self.scaling = alpha / rank

        # Quantized base layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self.quantizer = NF4Quantizer(block_size)
        self._quantized = False
        self._packed = None
        self._scales = None

        # LoRA matrices (full precision, trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def quantize_base(self):
        """Quantize the base linear layer to NF4."""
        self._packed, self._scales = self.quantizer.quantize(self.weight.data)
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining quantized base and LoRA.

        Steps:
        1. Dequantize base weights if quantized
        2. Compute base output: F.linear(x, weight, bias)
        3. Compute LoRA output: (x @ A^T @ B^T) * scaling
        4. Return base_output + lora_output
        """
        # Dequantize base weights if quantized
        if self._quantized:
            weight = self.quantizer.dequantize(self._packed, self._scales, self.weight.shape)
        else:
            weight = self.weight

        # Base output
        base_output = F.linear(x, weight, self.bias)

        # LoRA adaptation: (B @ A @ x) * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return base_output + lora_output


# =============================================================================
# Part 3: Apply QLoRA to Model
# =============================================================================

def apply_qlora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply QLoRA to a model: replace target linear layers with QLoRALinear.

    Steps:
    1. Iterate through model.named_modules()
    2. For each nn.Linear that matches target_modules:
       a. Create QLoRALinear with same dimensions
       b. Copy weights from original layer
       c. Quantize the base weights
       d. Replace the module in the model
    3. Return modified model

    Args:
        model: Model to adapt.
        rank: LoRA rank.
        alpha: LoRA alpha.
        target_modules: Target module names to replace.
        dropout: LoRA dropout.

    Returns:
        Model with QLoRA applied.
    """
    if target_modules is None:
        target_modules = ["W_q", "W_k", "W_v", "W_o", "w1", "w2", "w_gate"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name:
                    qlora_layer = QLoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=module.bias is not None,
                    )
                    qlora_layer.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        qlora_layer.bias.data = module.bias.data.clone()
                    qlora_layer.quantize_base()

                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model
                    for part in parent_name.split("."):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, child_name, qlora_layer)
                    break

    return model
