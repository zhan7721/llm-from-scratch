"""GRPO (Group Relative Policy Optimization) -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
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
    """Monte-Carlo approximation of KL divergence (k3 estimator)."""
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    """Compute mean of tensor, optionally respecting a mask."""
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages by normalizing rewards within each group.

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
    """
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + eps)
    return advantages


class GRPOLossSolution(nn.Module):
    """GRPO clipped surrogate objective with KL penalty.

    ratio = exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = clip(ratio, 1-eps, 1+eps) * advantages
    loss = -min(surr1, surr2) + kl_weight * KL
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
        """Compute the GRPO clipped loss."""
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2)

        loss = policy_loss + self.kl_weight * kl

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        kl_mean = masked_mean(kl, action_mask, dim=-1).mean()

        return loss, kl_mean


def grpo_step_solution(
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
    """Perform a single GRPO training step."""
    model.train()
    optimizer.zero_grad()

    logits = model(sequences)
    log_probs_all = F.log_softmax(logits, dim=-1)

    tokens = sequences[:, 1:].unsqueeze(-1)
    new_log_probs = log_probs_all[:, :-1, :].gather(-1, tokens).squeeze(-1)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1).expand_as(new_log_probs)

    grpo_loss = GRPOLossSolution(clip_eps=clip_eps, kl_weight=kl_weight)
    loss, kl = grpo_loss(
        log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        log_probs_ref=ref_log_probs,
        action_mask=action_mask,
    )

    loss.backward()

    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return {
        "loss": loss.item(),
        "kl": kl.item(),
    }
