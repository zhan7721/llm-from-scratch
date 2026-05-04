"""Direct Preference Optimization (DPO) -- Exercise.

Fill in the TODO methods to implement DPO components for language model
alignment. Start with `compute_log_probs`, then `DPOLoss`, and finally
`DPODataset.__init__` and `DPOTrainer.step`.

DPO has three key ideas:
1. No reward model needed -- the policy itself defines an implicit reward
2. The loss directly optimizes the policy on preference data
3. A frozen reference model provides the KL baseline

The implicit reward is: r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_start_idx: int,
) -> torch.Tensor:
    """Compute log probabilities of response tokens given a language model.

    For an autoregressive language model, logits at position t predict
    the token at position t+1. So the log prob of the response token
    at position t comes from logits at position t-1.

    Args:
        model: A language model returning logits (batch, seq, vocab_size).
        input_ids: Token IDs (batch, seq_len) = [prompt, response].
        response_start_idx: Index where response tokens begin.

    Returns:
        Sum of log probs over response tokens, shape (batch,).

    Hints:
        1. Get logits from the model: logits = model(input_ids)
        2. Compute log softmax over vocab: log_probs_all = F.log_softmax(logits, dim=-1)
        3. Response tokens: input_ids[:, response_start_idx:]
        4. Their log probs come from logits at position response_start_idx-1 to seq_len-2
        5. Use gather to select the log prob of each actual token
        6. Sum over the response dimension
    """
    raise NotImplementedError("TODO: implement compute_log_probs")


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss.

    The DPO loss is derived from the Bradley-Terry preference model
    combined with the optimal policy under a KL constraint:

        r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))

        loss = -log sigmoid(
            beta * (log pi(y_w|x) - log pi_ref(y_w|x))
            - beta * (log pi(y_l|x) - log pi_ref(y_l|x))
        )

    Hints:
        1. Compute log_ratio_chosen = policy_chosen - ref_chosen
        2. Compute log_ratio_rejected = policy_rejected - ref_rejected
        3. loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        4. Return the mean

    Args:
        beta: Temperature for implicit reward (typically 0.1-0.5).
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        ref_chosen_log_probs: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the DPO loss.

        Args:
            policy_chosen_log_probs: Log pi(y_w|x), shape (batch,).
            policy_rejected_log_probs: Log pi(y_l|x), shape (batch,).
            ref_chosen_log_probs: Log pi_ref(y_w|x), shape (batch,).
            ref_rejected_log_probs: Log pi_ref(y_l|x), shape (batch,).

        Returns:
            Scalar DPO loss (mean over batch).
        """
        raise NotImplementedError("TODO: implement DPOLoss.forward")


class DPODataset(Dataset):
    """Dataset for DPO preference pairs with pre-computed reference log probs.

    Each example has a prompt, a chosen (preferred) response, and a
    rejected response. During init, the reference model is used to
    pre-compute log pi_ref(y|x) for both responses.

    Args:
        pairs: List of dicts with "prompt", "chosen", "rejected" (token ID lists).
        ref_model: The reference (frozen) language model.
    """

    def __init__(
        self,
        pairs: List[Dict[str, List[int]]],
        ref_model: nn.Module,
    ) -> None:
        """Initialize DPO dataset.

        Hints:
            1. Freeze ref_model: set training=False, requires_grad=False for all params
            2. For each pair, concatenate prompt + chosen and prompt + rejected
            3. Use compute_log_probs with the ref_model to get reference log probs
            4. Store all tensors for later retrieval in __getitem__
        """
        raise NotImplementedError("TODO: implement DPODataset.__init__")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair with pre-computed reference log probs."""
        raise NotImplementedError("TODO: implement DPODataset.__getitem__")


class DPOTrainer:
    """DPO Trainer for language model alignment.

    Orchestrates the DPO training loop:
    1. Compute policy log probs for chosen and rejected
    2. Use pre-computed reference log probs from the dataset
    3. Compute DPO loss and update the policy

    Args:
        model: The policy model (being trained).
        ref_model: The reference model (frozen).
        beta: Temperature for DPO loss.
        lr: Learning rate.
        max_grad_norm: Max gradient norm for clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        lr: float = 5e-7,
        max_grad_norm: float = 1.0,
    ) -> None:
        """Initialize DPO trainer.

        Hints:
            1. Store model and ref_model
            2. Create DPOLoss with the given beta
            3. Create Adam optimizer for model parameters
            4. Freeze ref_model (no grad)
        """
        raise NotImplementedError("TODO: implement DPOTrainer.__init__")

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one DPO training step.

        Args:
            batch: Dict with chosen_input_ids, rejected_input_ids,
                   chosen_ref_log_probs, rejected_ref_log_probs,
                   response_start_idx.

        Returns:
            Dict with "loss" metric.

        Hints:
            1. Zero gradients
            2. Compute policy log probs using compute_log_probs
            3. Compute DPO loss
            4. Backward and step
            5. Return {"loss": loss.item()}
        """
        raise NotImplementedError("TODO: implement DPOTrainer.step")
