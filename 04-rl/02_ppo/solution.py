"""PPO (Proximal Policy Optimization) -- Reference Solution.

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


class PPOClipLossSolution(nn.Module):
    """PPO clipped surrogate objective with KL penalty.

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
        """Compute the PPO clipped loss."""
        # KL from reference model
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        # Probability ratio
        ratio = (log_probs - old_log_probs).exp()

        # Clipped surrogate objectives
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

        # Pessimistic bound: take the minimum
        policy_loss = -torch.min(surr1, surr2)

        # Add KL penalty
        loss = policy_loss + self.kl_weight * kl

        # Average over valid tokens and batch
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        kl_mean = masked_mean(kl, action_mask, dim=-1).mean()

        return loss, kl_mean


class GAESolution(nn.Module):
    """Generalized Advantage Estimation.

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = delta_t + (gamma * lambda) * A_{t+1}
    """

    def __init__(self, gamma: float, lam: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.lam = lam

    def forward(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        batch_size, seq_len = rewards.shape
        device = rewards.device

        advantages = torch.zeros_like(rewards)
        deltas = torch.zeros_like(rewards)

        # Compute TD errors
        for t in range(seq_len):
            if t == seq_len - 1:
                next_v = next_values
            else:
                next_v = values[:, t + 1]
            deltas[:, t] = rewards[:, t] + self.gamma * next_v - values[:, t]

        # Backward accumulation
        advantage = torch.zeros(batch_size, device=device)
        for t in reversed(range(seq_len)):
            advantage = deltas[:, t] + self.gamma * self.lam * advantage
            advantages[:, t] = advantage

        returns = advantages + values
        return advantages.detach(), returns.detach()


def rollout_solution(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    vocab_size: int,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Generate sequences from a policy and compute log probabilities."""
    batch_size = prompt.shape[0]
    device = prompt.device

    sequences = prompt.clone()
    all_log_probs = []
    all_values = []

    # Set model to eval mode for generation (disables dropout, etc.)
    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        logits, values = model(sequences)

        next_logits = logits[:, -1, :]  # (batch, vocab_size)
        next_value = values[:, -1]  # (batch,)

        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = next_logits.argmax(dim=-1)

        log_probs = F.log_softmax(next_logits, dim=-1)
        token_log_prob = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)

        all_log_probs.append(token_log_prob)
        all_values.append(next_value)

        sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)

    # Restore original training state
    if was_training:
        model.train()

    return {
        "sequences": sequences,
        "log_probs": torch.stack(all_log_probs, dim=1),
        "values": torch.stack(all_values, dim=1),
    }
