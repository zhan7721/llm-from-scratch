"""Proximal Policy Optimization (PPO) for language model alignment.

This module implements the core PPO components for RLHF:
- PPOClipLoss: clipped surrogate objective with KL penalty
- GAE: Generalized Advantage Estimation
- PPOTrainer: orchestrates policy update with old/new policy comparison
- rollout: generate responses and compute log probabilities

PPO is the workhorse algorithm for RLHF. It constrains policy updates
to stay close to the old policy, preventing catastrophic forgetting
while still allowing improvement.
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

    Uses the estimator from http://joschu.net/blog/kl-approx.html:
        KL ~ exp(log_ref - log_policy) - (log_ref - log_policy) - 1

    This is a per-token approximation. To get a scalar, average over tokens.

    Args:
        log_probs: Log probabilities from the current policy, shape (batch, seq).
        log_probs_ref: Log probabilities from the reference policy, shape (batch, seq).
        action_mask: Optional mask for valid tokens, shape (batch, seq).
                     1 for valid tokens, 0 for padding.

    Returns:
        Per-token KL divergence, shape (batch, seq).
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

    Args:
        tensor: Input tensor.
        mask: Binary mask (1 = include, 0 = exclude).
        dim: Dimension to average over.

    Returns:
        Masked mean.
    """
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


class PPOClipLoss(nn.Module):
    """PPO clipped surrogate objective with KL penalty.

    The PPO clip loss prevents large policy updates by clipping the
    probability ratio:

        ratio = exp(log_prob_new - log_prob_old)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1 - eps, 1 + eps) * advantages
        policy_loss = -min(surr1, surr2)

    The clipping mechanism ensures that when the advantage is positive,
    the ratio cannot exceed (1 + eps), and when the advantage is negative,
    the ratio cannot go below (1 - eps). This keeps the policy update
    conservative.

    Additionally, a KL divergence penalty from a reference model prevents
    the policy from diverging too far from the pretrained model:

        loss = policy_loss + kl_weight * KL(pi || pi_ref)

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
            log_probs: Log probabilities from the current policy, shape (batch, seq).
            old_log_probs: Log probabilities from the old policy, shape (batch, seq).
            advantages: Advantage estimates, shape (batch, seq).
            log_probs_ref: Log probabilities from the reference model, shape (batch, seq).
            action_mask: Optional mask for valid tokens, shape (batch, seq).

        Returns:
            Tuple of (loss, kl_divergence):
                - loss: Scalar PPO loss (to minimize).
                - kl_divergence: Mean KL divergence from reference (for monitoring).
        """
        # Step 1: Compute KL divergence from reference model
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        # Step 2: Compute probability ratio
        # ratio = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
        ratio = (log_probs - old_log_probs).exp()

        # Step 3: Compute surrogate objectives
        # surr1 = ratio * advantages (unclipped)
        surr1 = ratio * advantages
        # surr2 = clip(ratio, 1-eps, 1+eps) * advantages (clipped)
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

        # Step 4: Take the minimum (pessimistic bound)
        # -min(surr1, surr2) is the loss to minimize
        policy_loss = -torch.min(surr1, surr2)

        # Step 5: Add KL penalty
        loss = policy_loss + self.kl_weight * kl

        # Step 6: Average over valid tokens and batch
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        kl_mean = masked_mean(kl, action_mask, dim=-1).mean()

        return loss, kl_mean


class GAE(nn.Module):
    """Generalized Advantage Estimation (GAE).

    GAE computes advantage estimates that balance bias and variance
    using the parameter lambda:

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

    When lambda = 0: GAE reduces to one-step TD (high bias, low variance).
    When lambda = 1: GAE reduces to Monte Carlo (low bias, high variance).
    Typical values: lambda = 0.95, gamma = 0.99.

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
            next_values: Value estimate for the step after the last,
                        shape (batch,). This is V(s_{T+1}).

        Returns:
            Tuple of (advantages, returns):
                - advantages: GAE advantage estimates, shape (batch, seq).
                - returns: Discounted returns (advantages + values), shape (batch, seq).
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device

        # Initialize advantages tensor
        advantages = torch.zeros_like(rewards)

        # Compute TD errors (deltas) for each timestep
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # For the last timestep, V(s_{T+1}) is next_values
        deltas = torch.zeros_like(rewards)
        for t in range(seq_len):
            if t == seq_len - 1:
                # Last timestep: use next_values for V(s_{t+1})
                next_v = next_values
            else:
                # Non-last timestep: use values[:, t+1]
                next_v = values[:, t + 1]
            deltas[:, t] = rewards[:, t] + self.gamma * next_v - values[:, t]

        # Compute advantages by backward accumulation
        # A_t = delta_t + (gamma * lambda) * A_{t+1}
        advantage = torch.zeros(batch_size, device=device)
        for t in reversed(range(seq_len)):
            advantage = deltas[:, t] + self.gamma * self.lam * advantage
            advantages[:, t] = advantage

        # Returns = advantages + values
        returns = advantages + values

        return advantages.detach(), returns.detach()


class PPOTrainer:
    """PPO Trainer for language model alignment.

    Orchestrates the PPO training loop:
    1. Compute log probabilities for generated sequences
    2. Compute advantages using GAE
    3. Update policy using clipped surrogate loss
    4. Track KL divergence from reference model

    Args:
        model: The policy model (being trained).
        ref_model: The reference model (frozen, for KL penalty).
        clip_eps: PPO clipping parameter.
        kl_weight: Weight for KL penalty.
        gamma: Discount factor for GAE.
        lam: Lambda parameter for GAE.
        lr: Learning rate for the optimizer.
        ppo_epochs: Number of PPO update epochs per batch.
        max_grad_norm: Max gradient norm for gradient clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        clip_eps: float = 0.2,
        kl_weight: float = 0.1,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 1e-5,
        ppo_epochs: int = 4,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm

        self.ppo_loss = PPOClipLoss(clip_eps=clip_eps, kl_weight=kl_weight)
        self.gae = GAE(gamma=gamma, lam=lam)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _compute_log_probs(
        self,
        model: nn.Module,
        sequences: torch.Tensor,
        gen_start: int,
    ) -> torch.Tensor:
        """Compute log probabilities of generated tokens.

        Args:
            model: The model to compute log probs with.
            sequences: Full sequences (prompt + generated), shape (batch, seq).
            gen_start: Index where generated tokens start.

        Returns:
            Log probabilities of the generated tokens, shape (batch, gen_len).
        """
        is_ref = model is self.ref_model
        if is_ref:
            model.eval()

        logits, _ = model(sequences)

        # Log softmax over vocabulary
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Get log probs of the tokens that were actually generated
        # For each position t, we want log_prob of token at position gen_start + t
        # given everything before it (the logit at position gen_start + t - 1)
        gen_len = sequences.shape[1] - gen_start
        gen_log_probs = torch.zeros(sequences.shape[0], gen_len, device=sequences.device)

        for t in range(gen_len):
            pos = gen_start + t  # Position in the sequence
            if pos > 0:
                # Log prob of token at position 'pos', given everything before
                # Use gather for proper batch indexing
                token = sequences[:, pos].unsqueeze(1)  # (batch, 1)
                log_probs_at_pos = log_probs_all[:, pos - 1, :]  # (batch, vocab)
                gen_log_probs[:, t] = log_probs_at_pos.gather(1, token).squeeze(1)

        return gen_log_probs

    def step(
        self,
        sequences: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        prompt_len: int,
    ) -> Dict[str, float]:
        """Perform one PPO update step.

        Args:
            sequences: Full sequences (prompt + generated), shape (batch, seq).
            old_log_probs: Log probs from the old policy, shape (batch, gen_len).
            rewards: Rewards for each generated token, shape (batch, gen_len).
            values: Value estimates for each generated token, shape (batch, gen_len).
            prompt_len: Length of the prompt (to know where generation starts).

        Returns:
            Dict with training metrics (loss, kl, etc.).
        """
        gen_len = rewards.shape[1]

        # Compute next_values for GAE (value of the state after the last generated token)
        with torch.no_grad():
            _, next_values_full = self.model(sequences)
            # next_values is the value at the last generated position
            next_values = next_values_full[:, -1]

        # Compute advantages using GAE
        advantages, returns = self.gae(rewards, values, next_values)

        # Normalize advantages (standard practice in PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_kl = 0.0

        # PPO update epochs
        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad()

            # Compute current policy log probs
            new_log_probs = self._compute_log_probs(
                self.model, sequences, gen_start=prompt_len
            )

            # Compute reference policy log probs (detached, no grad)
            ref_log_probs = self._compute_log_probs(
                self.ref_model, sequences, gen_start=prompt_len
            )

            # Compute PPO loss
            loss, kl = self.ppo_loss(
                log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                log_probs_ref=ref_log_probs,
            )

            # Backward and optimize
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_kl += kl.item()

        return {
            "loss": total_loss / self.ppo_epochs,
            "kl": total_kl / self.ppo_epochs,
        }


def rollout(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    vocab_size: int,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Generate sequences from a policy and compute log probabilities.

    Autoregressively generates tokens from the model, collecting:
    - The full generated sequences
    - Log probabilities of each generated token
    - Value estimates at each generation step

    Args:
        model: The policy model (must return (logits, values)).
        prompt: Prompt token IDs, shape (batch, prompt_len).
        max_new_tokens: Number of tokens to generate.
        vocab_size: Vocabulary size (for bounds checking).
        temperature: Sampling temperature. Lower = more greedy.

    Returns:
        Dict with:
            - "sequences": Full sequences (prompt + generated), shape (batch, prompt_len + max_new_tokens).
            - "log_probs": Log probs of generated tokens, shape (batch, max_new_tokens).
            - "values": Value estimates at each step, shape (batch, max_new_tokens).
    """
    batch_size = prompt.shape[0]
    device = prompt.device

    # Start with the prompt
    sequences = prompt.clone()

    # Collect log probs and values for generated tokens
    all_log_probs = []
    all_values = []

    # Set model to eval mode for generation (disables dropout, etc.)
    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        # Forward pass
        logits, values = model(sequences)

        # Get logits and values for the last position
        next_logits = logits[:, -1, :]  # (batch, vocab_size)
        next_value = values[:, -1]  # (batch,)

        # Apply temperature
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            # Greedy decoding
            next_token = next_logits.argmax(dim=-1)

        # Compute log probability of the selected token
        log_probs = F.log_softmax(next_logits, dim=-1)
        token_log_prob = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)

        # Collect
        all_log_probs.append(token_log_prob)
        all_values.append(next_value)

        # Append generated token to sequence
        sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)

    # Restore original training state
    if was_training:
        model.train()

    return {
        "sequences": sequences,
        "log_probs": torch.stack(all_log_probs, dim=1),  # (batch, max_new_tokens)
        "values": torch.stack(all_values, dim=1),  # (batch, max_new_tokens)
    }
