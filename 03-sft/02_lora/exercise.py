"""LoRA implementation exercises.

Complete the TODO sections to implement Low-Rank Adaptation from scratch.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for a linear layer.

    Forward: y = x @ W + x @ A^T @ B^T * (alpha / rank)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original (frozen) linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False

        # TODO 1: Initialize lora_A as a Parameter of shape (rank, in_features)
        # Use nn.Parameter(torch.empty(rank, in_features))
        self.lora_A = ...

        # TODO 2: Initialize lora_B as a Parameter of shape (out_features, rank)
        # Initialize B with zeros (so LoRA starts as identity - no change to output)
        self.lora_B = ...

        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # TODO 3: Initialize lora_A with Kaiming uniform
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original output + LoRA adaptation."""
        # Original linear output
        orig_output = self.linear(x)

        # TODO 4: Compute LoRA output
        # 1. Apply dropout to x
        # 2. Multiply by lora_A.T
        # 3. Multiply by lora_B.T
        # 4. Scale by self.scaling
        lora_output = ...

        return orig_output + lora_output

    def merge(self):
        """Merge LoRA weights into the original linear layer."""
        # TODO 5: Add LoRA weights to the original weight matrix
        # self.linear.weight.data += (B @ A) * scaling
        ...

    def unmerge(self):
        """Remove LoRA weights from the original linear layer."""
        # TODO 6: Subtract LoRA weights from the original weight matrix
        ...


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA to specified linear layers in a model.

    Args:
        model: The model to adapt.
        rank: LoRA rank.
        alpha: LoRA alpha (scaling = alpha/rank).
        target_modules: List of module name patterns to target.
        dropout: LoRA dropout.

    Returns:
        The model with LoRA applied.
    """
    if target_modules is None:
        target_modules = ["W_q", "W_k", "W_v", "W_o", "w1", "w2", "w_gate"]

    # TODO 7: Iterate over all named modules
    # For each nn.Linear module whose name contains a target pattern:
    # 1. Create a LoRALinear with the same in/out features
    # 2. Copy the original weights
    # 3. Replace the module in the model
    ...

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA trainable parameters."""
    params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            params.append(param)
    return params


def count_lora_params(model: nn.Module) -> Tuple[int, int]:
    """Count LoRA vs total parameters.

    Returns:
        (lora_params, total_params)
    """
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    total_params = sum(p.numel() for p in model.parameters())
    return lora_params, total_params


def merge_lora(model: nn.Module):
    """Merge all LoRA layers in a model."""
    # TODO 8: Find all LoRALinear modules and call merge()
    ...


def unmerge_lora(model: nn.Module):
    """Unmerge all LoRA layers in a model."""
    # TODO 9: Find all LoRALinear modules and call unmerge()
    ...
