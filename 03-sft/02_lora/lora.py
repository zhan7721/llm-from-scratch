"""Low-Rank Adaptation (LoRA) implementation from scratch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class LoRALinear(nn.Module):
    """Low-Rank Adaptation for a linear layer.

    Instead of training the full weight matrix W (d_out x d_in),
    LoRA trains two small matrices:
        A: (d_in x rank)
        B: (rank x d_out)

    Forward: y = x @ (W + alpha/rank * B @ A)
    The original W is frozen; only A and B are trained.
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

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform (standard for LoRA)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original output + LoRA adaptation."""
        # Original linear output
        orig_output = self.linear(x)

        # LoRA adaptation: x @ A^T @ B^T * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return orig_output + lora_output

    def merge(self):
        """Merge LoRA weights into the original linear layer.

        After merging, the model can be used without the LoRA overhead.
        """
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    def unmerge(self):
        """Remove LoRA weights from the original linear layer."""
        self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling


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
                       Default: ["W_q", "W_k", "W_v", "W_o", "w1", "w2", "w_gate"]
        dropout: LoRA dropout.

    Returns:
        The model with LoRA applied.
    """
    if target_modules is None:
        target_modules = ["W_q", "W_k", "W_v", "W_o", "w1", "w2", "w_gate"]

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if this module name matches any target
            for target in target_modules:
                if target in name:
                    # Replace with LoRA version
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = model
                    for part in parent_name.split("."):
                        if part:
                            parent = getattr(parent, part)

                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=module.bias is not None,
                    )
                    # Copy original weights
                    lora_layer.linear.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_layer.linear.bias.data = module.bias.data.clone()

                    setattr(parent, child_name, lora_layer)
                    replaced += 1
                    break

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
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora(model: nn.Module):
    """Unmerge all LoRA layers in a model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
