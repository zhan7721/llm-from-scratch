"""PPO (Proximal Policy Optimization) -- Exercise.

Fill in the TODO methods to implement PPO components for RLHF.
Start with `PPOClipLoss`, then `GAE`, and finally `rollout`.

The PPO algorithm has three key ideas:
1. Clipped surrogate objective prevents large policy updates
2. GAE provides low-variance advantage estimates
3. KL penalty keeps policy close to reference model
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

    KL(pi || pi_ref) ~ exp(log_ref - log_pi) - (log_ref - log_pi) - 1

    This is already implemented for you -- use it in PPOClipLoss.
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

    This is already implemented for you -- use it in PPOClipLoss.
    """
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


class PPOClipLossExercise(nn.Module):
    """PPO clipped surrogate objective with KL penalty.

    The key insight: we want to maximize the expected advantage, but we
    clip the probability ratio to prevent the policy from changing too much.

    Mathematical formulation:
        ratio = pi_new(a|s) / pi_old(a|s)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-eps, 1+eps) * advantages
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
        """Compute the PPO clipped loss.

        Args:
            log_probs: Log probs from current policy, shape (batch, seq).
            old_log_probs: Log probs from old policy, shape (batch, seq).
            advantages: Advantage estimates, shape (batch, seq).
            log_probs_ref: Log probs from reference model, shape (batch, seq).
            action_mask: Optional mask for valid tokens, shape (batch, seq).

        Returns:
            Tuple of (loss, kl_divergence).
        """
        raise NotImplementedError("TODO: implement PPOClipLoss.forward")


class GAEExercise(nn.Module):
    """Generalized Advantage Estimation.

    GAE computes advantages by combining TD errors across timesteps:

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Then accumulates backwards:
        A_t = delta_t + (gamma * lambda) * A_{t+1}

    This balances bias (lambda=0, one-step TD) vs variance (lambda=1, Monte Carlo).

    Hints:
        1. Compute deltas: delta_t = r_t + gamma * next_V - V(s_t)
           - For t < T-1: next_V = values[:, t+1]
           - For t == T-1: next_V = next_values
        2. Backward accumulation: A_t = delta_t + gamma * lambda * A_{t+1}
           - Start from the end, work backwards
        3. returns = advantages + values
        4. Detach both before returning (GAE is a utility, not a loss)

    Args:
        gamma: Discount factor (typically 0.99).
        lam: GAE lambda parameter (typically 0.95).
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
        """Compute GAE advantages and returns.

        Args:
            rewards: Rewards at each timestep, shape (batch, seq).
            values: Value estimates at each timestep, shape (batch, seq).
            next_values: V(s_{T+1}), shape (batch,).

        Returns:
            Tuple of (advantages, returns), both shape (batch, seq).
        """
        raise NotImplementedError("TODO: implement GAE.forward")


def rollout_exercise(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    vocab_size: int,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Generate sequences from a policy and compute log probabilities.

    Autoregressively generates tokens from the model.

    Hints:
        1. Start with sequences = prompt.clone()
        2. For each generation step:
           a. Forward pass: logits, values = model(sequences)
           b. Get next_logits = logits[:, -1, :] and next_value = values[:, -1]
           c. If temperature > 0: sample from softmax(next_logits / temperature)
              Else: use argmax (greedy)
           d. Compute log_prob of the selected token using log_softmax + gather
           e. Append token to sequences
        3. Stack collected log_probs and values
        4. Return dict with "sequences", "log_probs", "values"

    Args:
        model: Policy model that returns (logits, values).
        prompt: Prompt token IDs, shape (batch, prompt_len).
        max_new_tokens: Number of tokens to generate.
        vocab_size: Vocabulary size.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        Dict with "sequences", "log_probs", "values".
    """
    raise NotImplementedError("TODO: implement rollout")
