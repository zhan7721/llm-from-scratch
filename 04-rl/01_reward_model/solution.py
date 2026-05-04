"""Reward Model -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class RewardModelSolution(nn.Module):
    """Wraps a transformer model and adds a scalar reward head.

    Architecture:
        input tokens -> transformer -> last token hidden state -> linear -> scalar reward
    """

    def __init__(self, transformer: nn.Module):
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
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute reward scores for input sequences.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Reward scores of shape (batch,).
        """
        # Get hidden states from the transformer
        hidden_states = self.transformer(input_ids)
        # Take the last token's hidden state (most context)
        last_hidden = hidden_states[:, -1, :]
        # Project to scalar reward and squeeze
        reward = self.scalar_head(last_hidden).squeeze(-1)
        return reward


class BradleyTerryLossSolution(nn.Module):
    """Bradley-Terry pairwise ranking loss for preference learning.

    P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    Loss = -log(sigmoid(r_chosen - r_rejected))
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Bradley-Terry pairwise ranking loss.

        Args:
            chosen_rewards: Rewards for preferred responses, shape (batch,).
            rejected_rewards: Rewards for rejected responses, shape (batch,).

        Returns:
            Scalar loss (mean over the batch).
        """
        # Use logsigmoid for numerical stability
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
        return loss.mean()


class RewardDatasetSolution(Dataset):
    """Dataset for preference pairs (chosen/rejected)."""

    def __init__(
        self,
        pairs: List[Dict[str, List[int]]],
        max_length: Optional[int] = None,
        pad_token_id: int = 0,
    ):
        self.pairs = pairs
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.pairs)

    def _pad_or_truncate(self, token_ids: List[int]) -> tuple:
        """Pad or truncate a sequence to max_length."""
        if self.max_length is None:
            ids = torch.tensor(token_ids, dtype=torch.long)
            mask = torch.ones(len(token_ids), dtype=torch.long)
            return ids, mask

        # Truncate if needed
        ids = token_ids[: self.max_length]
        seq_len = len(ids)

        # Pad if needed
        pad_len = self.max_length - seq_len
        if pad_len > 0:
            ids = ids + [self.pad_token_id] * pad_len

        ids_tensor = torch.tensor(ids, dtype=torch.long)
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])

        return ids_tensor, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair."""
        pair = self.pairs[idx]

        chosen_ids, chosen_mask = self._pad_or_truncate(pair["chosen"])
        rejected_ids, rejected_mask = self._pad_or_truncate(pair["rejected"])

        item = {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
        }

        if self.max_length is not None:
            item["chosen_attention_mask"] = chosen_mask
            item["rejected_attention_mask"] = rejected_mask

        return item


def train_reward_model_solution(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Single training step for the reward model.

    Args:
        model: The RewardModel to train.
        batch: Dict with "chosen_input_ids" and "rejected_input_ids".
        loss_fn: BradleyTerryLoss instance.
        optimizer: PyTorch optimizer.

    Returns:
        The loss tensor (scalar).
    """
    # Forward pass: compute rewards for chosen and rejected
    chosen_rewards = model(batch["chosen_input_ids"])
    rejected_rewards = model(batch["rejected_input_ids"])

    # Compute pairwise ranking loss
    loss = loss_fn(chosen_rewards, rejected_rewards)

    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
