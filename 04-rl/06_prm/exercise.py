"""Process Reward Model (PRM) -- Exercise.

Fill in the TODO methods to implement PRM components for step-level
reasoning evaluation.

Unlike Outcome Reward Models (ORRs) that only score the final answer,
PRMs score each reasoning step independently. This gives more granular
feedback -- you can identify exactly which step went wrong.

Architecture:
    input tokens -> transformer -> hidden states at step boundaries -> linear -> per-step logits

Start with `ProcessRewardModel.forward`, then `StepwiseRewardLoss.forward`,
and finally `best_of_n_prm`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class ProcessRewardModelExercise(nn.Module):
    """Scores each reasoning step independently using a transformer backbone.

    Takes a sequence of tokens (the full reasoning chain) and step boundary
    positions, and outputs a logit score for each step.

    Hints:
        1. Get hidden states: hidden_states = self.transformer(input_ids)
        2. Create batch indices for advanced indexing:
           batch_indices = torch.arange(batch_size, device=...)
           batch_indices = batch_indices.unsqueeze(1).expand_as(step_boundaries)
        3. Gather step hidden states: step_hidden = hidden_states[batch_indices, step_boundaries]
        4. Score each step: step_logits = self.step_head(step_hidden).squeeze(-1)

    Args:
        transformer: A transformer model that returns hidden states
                     (batch, seq_len, d_model) when given input_ids (batch, seq_len).
    """

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

        # Determine d_model from the transformer
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
        """Compute per-step reward scores (logits) for a batch of reasoning chains.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            step_boundaries: End positions of each reasoning step,
                             shape (batch, num_steps). LongTensor.

        Returns:
            Per-step logit scores of shape (batch, num_steps).
        """
        raise NotImplementedError("TODO: implement ProcessRewardModel.forward")


class StepwiseRewardDatasetExercise(Dataset):
    """Dataset for step-level reward training with per-step labels.

    Each example contains:
        - input_ids: the full tokenized reasoning chain
        - step_boundaries: end position of each reasoning step
        - step_labels: binary label per step (1 = correct, 0 = incorrect)

    Hints:
        1. Store the data list
        2. __len__ returns len(self.data)
        3. __getitem__ converts lists to tensors:
           - input_ids: torch.long
           - step_boundaries: torch.long
           - step_labels: torch.float

    Args:
        data: List of dicts with "input_ids", "step_boundaries", "step_labels".
    """

    def __init__(self, data: List[Dict[str, List[int]]]) -> None:
        raise NotImplementedError("TODO: implement StepwiseRewardDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError("TODO: implement StepwiseRewardDataset.__len__")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example.

        Returns:
            Dict with "input_ids" (long), "step_boundaries" (long), "step_labels" (float).
        """
        raise NotImplementedError("TODO: implement StepwiseRewardDataset.__getitem__")


class StepwiseRewardLossExercise(nn.Module):
    """Binary cross-entropy loss over step-level reward scores.

    Hints:
        1. Use F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
           to get per-step losses.
        2. If step_mask is provided, multiply per_step_loss by mask and
           divide by mask.sum() for the mean.
        3. Otherwise, just take per_step_loss.mean().
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        step_scores: torch.Tensor,
        step_labels: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute stepwise binary cross-entropy loss.

        Args:
            step_scores: Raw logits from PRM, shape (batch, num_steps).
            step_labels: Binary labels, shape (batch, num_steps).
            step_mask: Optional mask (1=valid, 0=padding), shape (batch, num_steps).

        Returns:
            Scalar loss.
        """
        raise NotImplementedError("TODO: implement StepwiseRewardLoss.forward")


def best_of_n_prm_exercise(
    step_scores: torch.Tensor,
    aggregation: str = 'min',
    step_mask: Optional[torch.Tensor] = None,
) -> int:
    """Select the best candidate by aggregating per-step PRM scores.

    Hints:
        1. For 'min': use step_scores.min(dim=1).values
           - With mask: replace masked positions with +inf before min
        2. For 'sum': use step_scores.sum(dim=1)
           - With mask: multiply by mask first
        3. For 'mean': use step_scores.mean(dim=1)
           - With mask: sum and divide by mask.sum(dim=1)
        4. For 'product': use step_scores.prod(dim=1)
           - With mask: replace masked positions with 1.0 before prod
        5. Return int(agg_scores.argmax().item())

    Args:
        step_scores: (n_candidates, max_steps) tensor of per-step scores.
        aggregation: 'min', 'sum', 'mean', or 'product'.
        step_mask: Optional (n_candidates, max_steps) mask.

    Returns:
        Index (int) of the best candidate.
    """
    raise NotImplementedError("TODO: implement best_of_n_prm")
