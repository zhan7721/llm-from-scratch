"""GRPO (Group Relative Policy Optimization) -- Exercise.

Fill in the TODO methods to implement GRPO components for RLHF.
Start with `compute_group_advantages`, then `GRPOLoss`, and finally `grpo_step`.

GRPO is a simpler alternative to PPO. The key insight:
- Generate G responses per prompt
- Normalize rewards within each group to get advantages
- No value network needed!

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

This means advantages always sum to zero within a group.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Monte-Carlo approximation of KL divergence (k3 estimator).

    This is already implemented for you -- use it in GRPOLoss.
    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    """Compute mean of tensor, optionally respecting a mask.

    This is already implemented for you -- use it in GRPOLoss.
    """
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages by normalizing rewards within each group.

    For GRPO, each prompt has G responses. The advantage of each response
    is computed relative to the other responses in the same group:

        advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

    Hints:
        1. Compute mean across the group dimension (dim=1, keepdim=True)
        2. Compute std across the group dimension (dim=1, keepdim=True)
        3. Normalize: (rewards - mean) / (std + eps)

    Args:
        rewards: Reward scores for each response, shape (num_prompts, G).
                 Each row is a group of G responses for one prompt.
        eps: Small constant for numerical stability in division.

    Returns:
        Group-relative advantages, shape (num_prompts, G).
        Each row sums to zero.
    """
    raise NotImplementedError("TODO: implement compute_group_advantages")


class GRPOLossExercise(nn.Module):
    """GRPO clipped surrogate objective with KL penalty.

    GRPO uses the same clipped surrogate as PPO, but with group-relative
    advantages instead of value-network advantages:

        ratio = exp(log_prob_new - log_prob_old)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1 - eps, 1 + eps) * advantages
        loss = -min(surr1, surr2) + kl_weight * KL

    Hints:
        1. Compute KL using approx_kl_divergence
        2. Compute ratio = exp(log_probs - old_log_probs)
        3. Compute surr1 = ratio * advantages
        4. Compute surr2 = clamp(ratio, 1-eps, 1+eps) * advantages
        5. policy_loss = -min(surr1, surr2)
        6. loss = policy_loss + kl_weight * kl
        7. Use masked_mean to average, return (loss, kl_mean)

    Args:
        clip_eps: Clipping parameter epsilon (typically 0.1-0.3).
        kl_weight: Weight for the KL penalty term.
    """

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_ref: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the GRPO clipped loss.

        Args:
            log_probs: Log probs from current policy, shape (batch, seq).
            old_log_probs: Log probs from old policy, shape (batch, seq).
            advantages: Group-relative advantages, shape (batch, seq).
            log_probs_ref: Log probs from reference model, shape (batch, seq).
            action_mask: Optional mask for valid tokens, shape (batch, seq).

        Returns:
            Tuple of (loss, kl_divergence).
        """
        raise NotImplementedError("TODO: implement GRPOLoss.forward")


def grpo_step_exercise(
    model: nn.Module,
    ref_model: nn.Module,
    sequences: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip_eps: float = 0.2,
    kl_weight: float = 0.1,
    max_grad_norm: float = 1.0,
    action_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Perform a single GRPO training step.

    Hints:
        1. Set model to training mode: model.train()
        2. Zero gradients: optimizer.zero_grad()
        3. Forward pass: logits = model(sequences)
        4. Compute log_probs: log_softmax + gather
        5. If advantages is 1D, expand to match log_probs shape
        6. Create GRPOLoss and compute loss
        7. Backward: loss.backward()
        8. Clip gradients if max_grad_norm > 0
        9. Optimizer step: optimizer.step()
        10. Return {"loss": loss.item(), "kl": kl.item()}

    Args:
        model: The policy model (being trained).
        ref_model: The reference model (frozen, for KL penalty).
        sequences: Full sequences, shape (batch, seq).
        old_log_probs: Log probs from old policy, shape (batch, seq-1).
        ref_log_probs: Log probs from reference model, shape (batch, seq-1).
        advantages: Group-relative advantages, shape (batch,) or (batch, seq-1).
        optimizer: Optimizer for the policy model.
        clip_eps: Clipping parameter.
        kl_weight: Weight for the KL penalty.
        max_grad_norm: Max gradient norm for gradient clipping.
        action_mask: Optional mask for valid tokens.

    Returns:
        Dict with training metrics ("loss", "kl").
    """
    raise NotImplementedError("TODO: implement grpo_step")
