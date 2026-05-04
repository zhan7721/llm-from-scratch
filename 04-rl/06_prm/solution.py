"""Process Reward Model (PRM) -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class ProcessRewardModelSolution(nn.Module):
    """Scores each reasoning step independently using a transformer backbone.

    Architecture:
        input_ids -> transformer -> hidden_states[step_boundaries] -> linear -> logits
    """

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

        d_model = getattr(transformer, "d_model", None)
        if d_model is None:
            d_model = getattr(transformer, "config", None)
            if d_model is not None:
                d_model = getattr(d_model, "d_model", None)
        if d_model is None:
            d_model = transformer.embedding.embedding_dim

        self.d_model = d_model
        self.step_head = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        step_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-step reward scores (logits)."""
        # Get hidden states from the transformer
        hidden_states = self.transformer(input_ids)  # (batch, seq_len, d_model)

        # Create batch indices for advanced indexing
        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        batch_indices = batch_indices.unsqueeze(1).expand_as(step_boundaries)

        # Gather hidden states at step boundary positions
        step_hidden = hidden_states[batch_indices, step_boundaries]  # (batch, num_steps, d_model)

        # Score each step
        step_logits = self.step_head(step_hidden).squeeze(-1)  # (batch, num_steps)
        return step_logits


class StepwiseRewardDatasetSolution(Dataset):
    """Dataset for step-level reward training with per-step labels."""

    def __init__(self, data: List[Dict[str, List[int]]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example with per-step labels."""
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "step_boundaries": torch.tensor(item["step_boundaries"], dtype=torch.long),
            "step_labels": torch.tensor(item["step_labels"], dtype=torch.float),
        }


class StepwiseRewardLossSolution(nn.Module):
    """Binary cross-entropy loss over step-level reward scores."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        step_scores: torch.Tensor,
        step_labels: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute stepwise binary cross-entropy loss."""
        per_step_loss = F.binary_cross_entropy_with_logits(
            step_scores, step_labels, reduction='none'
        )

        if step_mask is not None:
            per_step_loss = per_step_loss * step_mask
            return per_step_loss.sum() / step_mask.sum()
        else:
            return per_step_loss.mean()


def best_of_n_prm_solution(
    step_scores: torch.Tensor,
    aggregation: str = 'min',
    step_mask: Optional[torch.Tensor] = None,
) -> int:
    """Select the best candidate by aggregating per-step PRM scores."""
    if aggregation == 'min':
        if step_mask is not None:
            masked_scores = step_scores.masked_fill(step_mask == 0, float('inf'))
            agg_scores = masked_scores.min(dim=1).values
        else:
            agg_scores = step_scores.min(dim=1).values

    elif aggregation == 'sum':
        if step_mask is not None:
            agg_scores = (step_scores * step_mask).sum(dim=1)
        else:
            agg_scores = step_scores.sum(dim=1)

    elif aggregation == 'mean':
        if step_mask is not None:
            agg_scores = (step_scores * step_mask).sum(dim=1) / step_mask.sum(dim=1)
        else:
            agg_scores = step_scores.mean(dim=1)

    elif aggregation == 'product':
        if step_mask is not None:
            masked_scores = step_scores.masked_fill(step_mask == 0, 1.0)
            agg_scores = masked_scores.prod(dim=1)
        else:
            agg_scores = step_scores.prod(dim=1)

    else:
        raise ValueError(
            f"Unknown aggregation method: {aggregation}. "
            f"Choose from 'min', 'sum', 'mean', 'product'."
        )

    return int(agg_scores.argmax().item())
