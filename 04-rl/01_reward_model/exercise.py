"""Reward Model -- Exercise.

Fill in the TODO methods to implement reward model components.
Start with `BradleyTerryLoss`, then `RewardModel`, and finally
`RewardDataset.__getitem__` and `train_reward_model`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class RewardModelExercise(nn.Module):
    """Wraps a transformer model and adds a scalar reward head.

    The reward model takes a transformer (which produces hidden states of
    shape (batch, seq, d_model)) and adds a linear projection to produce
    a single scalar reward score per input sequence.

    Architecture:
        input tokens -> transformer -> last token hidden state -> linear -> scalar reward

    Args:
        transformer: A transformer model that returns hidden states
                     (batch, seq, d_model) when given input_ids (batch, seq).
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

        # Scalar head: projects the last hidden state to a single reward value
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute reward scores for input sequences.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Reward scores of shape (batch,). One scalar per sequence.

        Hints:
            1. Pass input_ids through self.transformer to get hidden states.
            2. Take the last token's hidden state: hidden_states[:, -1, :]
            3. Pass through self.scalar_head.
            4. Squeeze the last dimension to get shape (batch,).
        """
        raise NotImplementedError("TODO: implement forward")


class BradleyTerryLossExercise(nn.Module):
    """Bradley-Terry pairwise ranking loss for preference learning.

    The Bradley-Terry model:
        P(chosen > rejected) = sigmoid(r_chosen - r_rejected)

    Loss = -log(sigmoid(r_chosen - r_rejected))

    Hints:
        - Use F.logsigmoid for numerical stability.
        - Return the mean loss over the batch.
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

        Hints:
            1. Compute the difference: chosen_rewards - rejected_rewards.
            2. Apply F.logsigmoid to get log(sigmoid(diff)).
            3. Negate it: -logsigmoid(diff).
            4. Return the mean.
        """
        raise NotImplementedError("TODO: implement BradleyTerryLoss.forward")


class RewardDatasetExercise(Dataset):
    """Dataset for preference pairs (chosen/rejected).

    Each example is a dict with:
        - "chosen": list of token IDs for the preferred response
        - "rejected": list of token IDs for the rejected response

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair.

        Args:
            idx: Index into the pairs list.

        Returns:
            Dict with "chosen_input_ids" and "rejected_input_ids" tensors.
            If max_length is set, also includes attention masks.

        Hints:
            1. Get the pair at index idx.
            2. Convert pair["chosen"] and pair["rejected"] to tensors.
            3. If self.max_length is set:
               - Truncate sequences longer than max_length.
               - Pad shorter sequences with pad_token_id.
               - Create attention masks (1 for real tokens, 0 for padding).
            4. Return a dict with the tensors.
        """
        raise NotImplementedError("TODO: implement RewardDataset.__getitem__")


def train_reward_model_exercise(
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

    Hints:
            1. Compute rewards for chosen: model(batch["chosen_input_ids"])
            2. Compute rewards for rejected: model(batch["rejected_input_ids"])
            3. Compute loss: loss_fn(chosen_rewards, rejected_rewards)
            4. Zero gradients, backward, step.
            5. Return the loss.
    """
    raise NotImplementedError("TODO: implement train_reward_model")
