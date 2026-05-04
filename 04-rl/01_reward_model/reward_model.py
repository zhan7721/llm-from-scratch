"""Reward Model for Reinforcement Learning from Human Feedback (RLHF).

This module implements the core components for training a reward model:
- RewardModel: wraps a transformer, adds a scalar head for reward scoring
- BradleyTerryLoss: pairwise ranking loss for preference learning
- RewardDataset: loads preference pairs (chosen/rejected)
- train_reward_model: training loop for the reward model

The reward model is the bridge between human preferences and RL training.
It learns to score text quality by training on pairs of (chosen, rejected)
responses using the Bradley-Terry preference model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class RewardModel(nn.Module):
    """Wraps a transformer model and adds a scalar reward head.

    The reward model takes a transformer (which produces hidden states of
    shape (batch, seq, d_model)) and adds a linear projection to produce
    a single scalar reward score per input sequence.

    Architecture:
        input tokens -> transformer -> last token hidden state -> linear -> scalar reward

    We use the last token's hidden state because in autoregressive models,
    the last token has attended to all previous tokens and thus contains
    the most complete representation of the sequence.

    Args:
        transformer: A transformer model that returns hidden states
                     (batch, seq, d_model) when given input_ids (batch, seq).
    """

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

        # Determine d_model from the transformer
        # Common attribute names for hidden dimension
        d_model = getattr(transformer, "d_model", None)
        if d_model is None:
            d_model = getattr(transformer, "config", None)
            if d_model is not None:
                d_model = getattr(d_model, "d_model", None)
        if d_model is None:
            # Try to infer from embedding dimension
            d_model = transformer.embedding.embedding_dim

        self.d_model = d_model

        # Scalar head: projects the last hidden state to a single reward value
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute reward scores for input sequences.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Reward scores of shape (batch,). One scalar per sequence.
        """
        # Step 1: Get hidden states from the transformer
        # Shape: (batch, seq_len, d_model)
        hidden_states = self.transformer(input_ids)

        # Step 2: Take the last token's hidden state
        # In autoregressive models, the last token has the most context
        # Shape: (batch, d_model)
        last_hidden = hidden_states[:, -1, :]

        # Step 3: Project to scalar reward
        # Shape: (batch, 1) -> squeeze to (batch,)
        reward = self.scalar_head(last_hidden).squeeze(-1)

        return reward


class BradleyTerryLoss(nn.Module):
    """Bradley-Terry pairwise ranking loss for preference learning.

    The Bradley-Terry model defines the probability that response A is
    preferred over response B as:

        P(A > B) = sigmoid(r_A - r_B)

    The loss is the negative log-likelihood:

        loss = -log(sigmoid(r_chosen - r_rejected))

    This loss encourages the reward model to assign higher scores to
    chosen (preferred) responses than to rejected responses.

    When r_chosen >> r_rejected, sigmoid approaches 1 and loss approaches 0.
    When r_chosen << r_rejected, sigmoid approaches 0 and loss is very high.
    When r_chosen == r_rejected, loss = log(2) (maximum uncertainty).
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
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        # We use logsigmoid for numerical stability:
        # -log(sigmoid(x)) = -logsigmoid(x)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
        return loss.mean()


class RewardDataset(Dataset):
    """Dataset for preference pairs (chosen/rejected).

    Each example is a dict with:
        - "chosen": list of token IDs for the preferred response
        - "rejected": list of token IDs for the rejected response

    Supports optional padding to a fixed max_length.

    Args:
        pairs: List of dicts, each with "chosen" and "rejected" token ID lists.
        max_length: If set, pad/truncate all sequences to this length.
        pad_token_id: Token ID to use for padding (default: 0).
    """

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

    def _pad_or_truncate(
        self, token_ids: List[int]
    ) -> tuple:
        """Pad or truncate a sequence to max_length.

        Args:
            token_ids: List of token IDs.

        Returns:
            Tuple of (padded_ids, attention_mask) as tensors.
        """
        if self.max_length is None:
            # No padding: just convert to tensor
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
        # Attention mask: 1 for real tokens, 0 for padding
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])

        return ids_tensor, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair.

        Args:
            idx: Index into the pairs list.

        Returns:
            Dict with:
                - "chosen_input_ids": tensor of chosen token IDs
                - "rejected_input_ids": tensor of rejected token IDs
                - "chosen_attention_mask": mask (if padding enabled)
                - "rejected_attention_mask": mask (if padding enabled)
        """
        pair = self.pairs[idx]

        chosen_ids, chosen_mask = self._pad_or_truncate(pair["chosen"])
        rejected_ids, rejected_mask = self._pad_or_truncate(pair["rejected"])

        item = {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
        }

        # Include attention masks when padding is used
        if self.max_length is not None:
            item["chosen_attention_mask"] = chosen_mask
            item["rejected_attention_mask"] = rejected_mask

        return item


def train_reward_model(
    model: RewardModel,
    batch: Dict[str, torch.Tensor],
    loss_fn: BradleyTerryLoss,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Single training step for the reward model.

    Computes rewards for chosen and rejected responses, then computes
    the Bradley-Terry loss and performs a gradient update.

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
