"""Mixture of Experts (MoE) — Exercise.

Fill in the TODO methods to implement a working MoE layer.
Start with the Expert, then the Router, then the full MoE layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouterExercise(nn.Module):
    """Routes each token to the top-K experts based on a learned gating network.

    Args:
        d_model: Dimensionality of the input features.
        n_experts: Total number of experts.
        top_k: Number of experts to route each token to.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k

        # TODO: Create a linear gate that maps d_model -> n_experts (no bias)
        # HINT: self.gate = nn.Linear(d_model, n_experts, bias=False)
        pass

    def forward(self, x: torch.Tensor):
        """Route tokens to top-K experts.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            indices: (batch * seq_len, top_k) — expert indices for each token.
            weights: (batch * seq_len, top_k) — routing weights (softmax).

        Steps:
            1. Flatten x from (B, T, D) to (B*T, D)
            2. Compute logits = self.gate(x_flat)
            3. Use torch.topk to get the top-K values and indices
            4. Apply softmax to the top-K values to get routing weights
            5. Return indices and weights
        """
        raise NotImplementedError("TODO: implement forward")


class ExpertExercise(nn.Module):
    """Single expert: a SwiGLU-style feed-forward network.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Hidden dimensionality.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # TODO: Create three linear layers (all without bias):
        #   self.w1      — maps d_model -> d_ff
        #   self.w_gate  — maps d_model -> d_ff (for the gating mechanism)
        #   self.w2      — maps d_ff -> d_model
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using SwiGLU activation.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model).

        Formula:
            output = w2(SiLU(w_gate(x)) * w1(x))

        Steps:
            1. Compute gate = F.silu(self.w_gate(x))
            2. Compute hidden = self.w1(x)
            3. Return self.w2(gate * hidden)
        """
        raise NotImplementedError("TODO: implement forward")


class MoELayerExercise(nn.Module):
    """Mixture of Experts layer.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Hidden dimensionality of each expert.
        n_experts: Total number of experts.
        top_k: Number of experts activated per token.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k

        # TODO: Create the router and expert list
        # HINT:
        #   self.router = TopKRouterExercise(d_model, n_experts, top_k)
        #   self.experts = nn.ModuleList([ExpertExercise(d_model, d_ff) for _ in range(n_experts)])
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).

        Steps:
            1. Get B, T, D from x.shape
            2. Get routing indices and weights from self.router(x)
            3. Flatten x to (B*T, D)
            4. Create a zero output tensor of the same shape
            5. For each of the top_k slots:
               a. Get the expert indices and weights for this slot
               b. For each expert e_idx in range(n_experts):
                  - Find which tokens are routed to this expert (mask)
                  - If any tokens match, run them through the expert
                  - Add the weighted expert output to the corresponding positions
            6. Reshape output back to (B, T, D) and return
        """
        raise NotImplementedError("TODO: implement forward")
