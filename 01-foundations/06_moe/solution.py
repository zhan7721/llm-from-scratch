"""Mixture of Experts (MoE) — Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouterSolution(nn.Module):
    """Routes each token to the top-K experts — solution version.

    Args:
        d_model: Dimensionality of the input features.
        n_experts: Total number of experts.
        top_k: Number of experts to route each token to.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """Route tokens to top-K experts.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            indices: (batch * seq_len, top_k) — expert indices for each token.
            weights: (batch * seq_len, top_k) — routing weights (softmax).
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        logits = self.gate(x_flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        return indices, weights


class ExpertSolution(nn.Module):
    """Single expert using SwiGLU activation — solution version.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Hidden dimensionality.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: output = w2(SiLU(w_gate(x)) * w1(x))."""
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))


class MoELayerSolution(nn.Module):
    """Mixture of Experts layer — solution version.

    Args:
        d_model: Input and output dimensionality.
        d_ff: Hidden dimensionality of each expert.
        n_experts: Total number of experts.
        top_k: Number of experts activated per token.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.router = TopKRouterSolution(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([ExpertSolution(d_model, d_ff) for _ in range(n_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, D = x.shape
        indices, weights = self.router(x)

        x_flat = x.view(-1, D)
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            expert_w = weights[:, k].unsqueeze(-1)

            for e_idx in range(len(self.experts)):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += expert_w[mask] * expert_output

        return output.view(B, T, D)
