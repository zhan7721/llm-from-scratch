"""Online DPO -- Exercise.

Fill in the TODO methods to implement Online DPO components. Online DPO
generates preference pairs on-the-fly from the current policy, rather than
using a static dataset.

Key ideas:
1. Generate multiple candidate responses from the current policy
2. Score candidates with a reward model to create preference pairs
3. Train with standard DPO loss on the fresh pairs
4. Repeat each epoch with new data from the updated policy

The advantage over offline DPO: no distribution mismatch between the
data-generating policy and the policy being trained.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

# Import DPO components from the previous chapter
sys.path.append(os.path.join(os.path.dirname(__file__), "../03_dpo"))
from dpo import compute_log_probs, DPOLoss


def generate_and_score(
    policy: nn.Module,
    reward_model: nn.Module,
    prompt: List[int],
    max_new_tokens: int = 32,
    num_candidates: int = 4,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Generate multiple responses from the policy, score with reward model,
    and return the best (chosen) and worst (rejected) as a preference pair.

    Process:
    1. Generate `num_candidates` responses using temperature sampling
    2. Score each complete sequence with the reward model
    3. Select highest-scored as "chosen", lowest as "rejected"
    4. Compute reference log probs for both

    Args:
        policy: The current policy model.
        reward_model: A reward model that scores sequences.
        prompt: Token IDs for the prompt.
        max_new_tokens: Number of tokens to generate per response.
        num_candidates: Number of candidate responses to generate.
        temperature: Sampling temperature.

    Returns:
        Dict with chosen_input_ids, rejected_input_ids, chosen_reward,
        rejected_reward, chosen_ref_log_probs, rejected_ref_log_probs,
        response_start_idx.

    Hints:
        1. Set models to eval mode
        2. For each candidate, start with the prompt and generate tokens
           autoregressively using torch.multinomial on softmax(logits/temp)
        3. Stack candidates into a batch and score with reward model
        4. Use argmax/argmin on rewards to select chosen/rejected
        5. Compute reference log probs using compute_log_probs
    """
    raise NotImplementedError("TODO: implement generate_and_score")


class OnlineDPODataset(Dataset):
    """Dataset that generates preference pairs on-the-fly.

    Generates preference pairs by running the policy to produce candidate
    responses and scoring them with a reward model. Can be refreshed to
    get new pairs from the updated policy.

    Args:
        policy: The current policy model.
        reward_model: Reward model for scoring responses.
        prompts: List of prompt token ID lists.
        max_new_tokens: Max tokens to generate per response.
        num_candidates: Number of candidates per prompt.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        policy: nn.Module,
        reward_model: nn.Module,
        prompts: List[List[int]],
        max_new_tokens: int = 32,
        num_candidates: int = 4,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the online DPO dataset.

        Hints:
            1. Store all parameters
            2. Freeze the reward model (eval mode, no grad)
            3. Generate initial preference pairs by calling refresh()
        """
        raise NotImplementedError("TODO: implement OnlineDPODataset.__init__")

    def refresh(self) -> None:
        """Regenerate all preference pairs from the current policy.

        Hints:
            1. Clear existing pairs
            2. For each prompt, call generate_and_score to get a new pair
            3. Store the pairs for later retrieval
        """
        raise NotImplementedError("TODO: implement OnlineDPODataset.refresh")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair with pre-computed reference log probs.

        Reference log probs are computed at generation time and stored,
        so this just returns the stored values.

        Returns:
            Dict with chosen_input_ids, rejected_input_ids,
            chosen_ref_log_probs, rejected_ref_log_probs, response_start_idx.
        """
        pair = self._pairs[idx]
        return {
            "chosen_input_ids": pair["chosen_input_ids"],
            "rejected_input_ids": pair["rejected_input_ids"],
            "chosen_ref_log_probs": pair["chosen_ref_log_probs"],
            "rejected_ref_log_probs": pair["rejected_ref_log_probs"],
            "response_start_idx": pair["response_start_idx"],
        }


class OnlineDPOTrainer:
    """Online DPO Trainer combining generation and training.

    Generates fresh preference data each epoch, then trains with DPO loss.

    Args:
        model: The policy model (being trained).
        ref_model: The reference model (frozen).
        reward_model: Reward model for scoring responses.
        prompts: List of prompt token ID lists.
        beta: Temperature for DPO loss.
        lr: Learning rate.
        max_grad_norm: Max gradient norm for clipping.
        max_new_tokens: Max tokens to generate per response.
        num_candidates: Number of candidates per prompt.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        prompts: List[List[int]],
        beta: float = 0.1,
        lr: float = 5e-7,
        max_grad_norm: float = 1.0,
        max_new_tokens: int = 32,
        num_candidates: int = 4,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the Online DPO trainer.

        Hints:
            1. Store model, ref_model, reward_model, and hyperparameters
            2. Create DPOLoss with the given beta
            3. Create Adam optimizer for model parameters
            4. Freeze ref_model and reward_model
        """
        raise NotImplementedError("TODO: implement OnlineDPOTrainer.__init__")

    def generate(self) -> OnlineDPODataset:
        """Generate fresh preference pairs from the current policy.

        Returns:
            OnlineDPODataset with fresh preference pairs.
        """
        raise NotImplementedError("TODO: implement OnlineDPOTrainer.generate")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
        raise NotImplementedError("TODO: implement OnlineDPOTrainer.train_step")

    def train_epoch(self, batch_size: int = 4) -> Dict[str, float]:
        """Run one full epoch of online DPO training.

        1. Generate fresh preference pairs
        2. Train on all pairs in mini-batches
        3. Return average metrics

        Args:
            batch_size: Number of preference pairs per mini-batch.

        Returns:
            Dict with average training metrics.
        """
        raise NotImplementedError("TODO: implement OnlineDPOTrainer.train_epoch")
