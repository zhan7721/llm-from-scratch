"""Process Reward Model (PRM) for step-level reasoning evaluation.

This module implements the core components for training and using a
Process Reward Model:
- ProcessRewardModel: transformer backbone + per-step linear head
- StepwiseRewardDataset: data with per-step labels (correct/incorrect)
- StepwiseRewardLoss: binary cross-entropy over step-level scores
- best_of_n_prm: select best response by aggregating step scores

Unlike Outcome Reward Models (ORMs) that only score the final answer,
PRMs score each reasoning step independently. This gives more granular
feedback for training reasoning models -- you can identify exactly which
step went wrong, not just whether the final answer is correct.

Architecture:
    input tokens -> transformer -> hidden states at step boundaries -> linear -> per-step logits

The step boundaries indicate where each reasoning step ends in the token
sequence. The hidden state at each boundary captures the full context up
to that point, making it a natural representation for scoring that step.

Reference: "Let's Verify Step by Step" (Lightman et al., 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class ProcessRewardModel(nn.Module):
    """Scores each reasoning step independently using a transformer backbone.

    Takes a sequence of tokens (the full reasoning chain) and step boundary
    positions, and outputs a logit score for each step. Higher scores
    indicate the step is more likely to be correct.

    Architecture:
        input_ids -> transformer -> hidden_states[step_boundaries] -> linear -> logits

    The hidden state at each step's end position is used because in
    autoregressive models, the last token of a step has attended to all
    tokens in that step (and all previous steps), giving a complete
    representation of the reasoning up to that point.

    Args:
        transformer: A transformer model that returns hidden states
                     (batch, seq_len, d_model) when given input_ids (batch, seq_len).
                     Must have a `d_model` attribute or inferable hidden dimension.
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
            # Try to infer from embedding dimension
            d_model = transformer.embedding.embedding_dim

        self.d_model = d_model

        # Per-step scoring head: maps each step's hidden state to a scalar logit
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
                             shape (batch, num_steps). Each value is an
                             index into the sequence indicating where that
                             step ends. Must be a LongTensor.
                             Example: [[3, 7, 11], [4, 8, 11]] means
                             batch item 0 has 3 steps ending at positions
                             3, 7, 11.

        Returns:
            Per-step logit scores of shape (batch, num_steps).
            Apply sigmoid to get probabilities in [0, 1].
        """
        # Step 1: Get hidden states from the transformer
        # Shape: (batch, seq_len, d_model)
        hidden_states = self.transformer(input_ids)

        # Step 2: Extract hidden states at step boundary positions
        # We use advanced indexing to gather the hidden state at the end
        # of each step for each batch item.
        batch_size = input_ids.shape[0]
        num_steps = step_boundaries.shape[1]

        # Create batch indices: [[0, 0, 0], [1, 1, 1], ...]
        # This pairs each batch item with its step positions
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        batch_indices = batch_indices.unsqueeze(1).expand_as(step_boundaries)

        # Gather: hidden_states[batch_i, step_end_j] for all i, j
        # Shape: (batch, num_steps, d_model)
        step_hidden = hidden_states[batch_indices, step_boundaries]

        # Step 3: Score each step with the linear head
        # Shape: (batch, num_steps, 1) -> squeeze -> (batch, num_steps)
        step_logits = self.step_head(step_hidden).squeeze(-1)

        return step_logits


class StepwiseRewardDataset(Dataset):
    """Dataset for step-level reward training with per-step labels.

    Each example contains:
        - input_ids: the full tokenized reasoning chain (all steps concatenated)
        - step_boundaries: end position of each reasoning step
        - step_labels: binary label per step (1 = correct, 0 = incorrect)

    The step_boundaries are indices into input_ids indicating where each
    step ends. For example, if a reasoning chain has steps of lengths
    4, 3, 5 tokens, the boundaries would be [3, 6, 11].

    Args:
        data: List of dicts, each with:
            - "input_ids": list of token IDs for the full reasoning chain
            - "step_boundaries": list of end positions for each step
            - "step_labels": list of 0/1 labels for each step
    """

    def __init__(self, data: List[Dict[str, List[int]]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example with per-step labels.

        Args:
            idx: Index into the data list.

        Returns:
            Dict with:
                - "input_ids": (seq_len,) long tensor of token IDs
                - "step_boundaries": (num_steps,) long tensor of end positions
                - "step_labels": (num_steps,) float tensor of 0/1 labels
        """
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "step_boundaries": torch.tensor(item["step_boundaries"], dtype=torch.long),
            "step_labels": torch.tensor(item["step_labels"], dtype=torch.float),
        }


class StepwiseRewardLoss(nn.Module):
    """Binary cross-entropy loss over step-level reward scores.

    Applies BCE loss (with logits) independently to each step's score
    against its binary label. This treats each step as an independent
    binary classification problem: is this step correct or not?

    The loss is averaged over all steps (and all batch items). If a
    step_mask is provided, masked steps are excluded from the average.

    This is numerically equivalent to:
        sigmoid(scores) -> binary_cross_entropy -> mean

    but uses the fused BCE-with-logits for better numerical stability.
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
            step_scores: Raw logit scores from the PRM, shape (batch, num_steps).
            step_labels: Binary labels (1=correct, 0=incorrect), shape (batch, num_steps).
            step_mask: Optional mask (1=valid, 0=padding), shape (batch, num_steps).
                       If provided, masked steps are excluded from the loss.

        Returns:
            Scalar loss (mean over valid steps).
        """
        # Compute per-step BCE loss (reduction='none' to keep per-step losses)
        # Shape: (batch, num_steps)
        per_step_loss = F.binary_cross_entropy_with_logits(
            step_scores, step_labels, reduction='none'
        )

        if step_mask is not None:
            # Zero out loss for masked steps, then average over valid steps
            per_step_loss = per_step_loss * step_mask
            return per_step_loss.sum() / step_mask.sum()
        else:
            # Average over all steps
            return per_step_loss.mean()


def best_of_n_prm(
    step_scores: torch.Tensor,
    aggregation: str = 'min',
    step_mask: Optional[torch.Tensor] = None,
) -> int:
    """Select the best candidate response by aggregating per-step PRM scores.

    Given N candidate responses, each scored per-step by a PRM, this function
    aggregates the step scores into a single score per candidate and returns
    the index of the best one.

    Aggregation methods:
        - 'min': Worst-case step score. Conservative: if any step is bad,
          the whole response is penalized. Good for ensuring all steps are correct.
        - 'sum': Total score across steps. Favors responses with more correct steps.
        - 'mean': Average step score. Normalizes for response length.
        - 'product': Product of step scores. Treats scores as probabilities
          of correctness (assumes independence). The probability that ALL
          steps are correct. Scores should be in [0, 1] for this to be meaningful.

    Args:
        step_scores: Per-step scores for each candidate,
                     shape (n_candidates, max_steps).
                     Can be probabilities (for product) or any real-valued scores.
        aggregation: Method to aggregate step scores.
                     One of 'min', 'sum', 'mean', 'product'.
        step_mask: Optional mask for valid steps (1=valid, 0=padding),
                   shape (n_candidates, max_steps).

    Returns:
        Index (int) of the best candidate.
    """
    n_candidates = step_scores.shape[0]

    if aggregation == 'min':
        if step_mask is not None:
            # Replace masked positions with +inf so they don't affect the min
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
            # Replace masked positions with 1.0 (identity for multiplication)
            masked_scores = step_scores.masked_fill(step_mask == 0, 1.0)
            agg_scores = masked_scores.prod(dim=1)
        else:
            agg_scores = step_scores.prod(dim=1)

    else:
        raise ValueError(
            f"Unknown aggregation method: {aggregation}. "
            f"Choose from 'min', 'sum', 'mean', 'product'."
        )

    # Return the index of the candidate with the highest aggregated score
    return int(agg_scores.argmax().item())
