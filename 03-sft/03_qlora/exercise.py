"""QLoRA Exercise: Implement Quantized Low-Rank Adaptation.

Complete the TODO sections to implement QLoRA from scratch.
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
        # TODO: Implement NF4 quantization
        # Hint: Use self.NF4_LEVELS as the target quantization levels
        # Hint: Use torch.argmin to find nearest level
        # Hint: Pack pairs of indices: packed = indices[:, 0::2] | (indices[:, 1::2] << 4)
        raise NotImplementedError("Implement NF4 quantize")

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
        # TODO: Compute scaling factor (alpha / rank)
        self.scaling = None  # TODO

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
        # TODO: Initialize self.lora_A
        raise NotImplementedError("Initialize lora_A")

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
        # TODO: Implement forward pass
        raise NotImplementedError("Implement QLoRA forward")


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

    # TODO: Implement apply_qlora
    raise NotImplementedError("Implement apply_qlora")
