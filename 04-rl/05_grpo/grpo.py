"""Group Relative Policy Optimization (GRPO) for language model alignment.

This module implements the core GRPO components:
- compute_group_advantages: normalize rewards within a group of responses
- GRPOLoss: clipped surrogate objective with KL penalty (like PPO, but
  uses group-relative advantages instead of value-network advantages)
- GRPOTrainer: generates multiple responses per prompt, scores them,
  and updates the policy using group-relative advantages
- grpo_step: a single training step (convenience function)

GRPO is a simpler alternative to PPO for RLHF. The key insight:
instead of training a value network to estimate advantages, generate
G responses per prompt and normalize rewards within the group.

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

This means advantages always sum to zero within a group, and the
relative ranking of responses determines the advantage signal.

Reference: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models" (Shao et al., 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Optional, Tuple


# ============================================================
# Helper functions (shared with PPO module)
# ============================================================


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


# ============================================================
# Core GRPO components
# ============================================================


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages by normalizing rewards within each group.

    For GRPO, each prompt has G responses. The advantage of each response
    is computed relative to the other responses in the same group:

        advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

    This is the key difference from PPO: instead of using a learned value
    function, we use the empirical mean and std of the group's rewards.

    By construction, advantages sum to zero within each group, and the
    relative ranking of responses determines the training signal.

    Args:
        rewards: Reward scores for each response, shape (num_prompts, G).
                 Each row is a group of G responses for one prompt.
        eps: Small constant for numerical stability in division.

    Returns:
        Group-relative advantages, shape (num_prompts, G).
        Each row sums to zero.
    """
    # Mean and std across the group dimension (dim=1)
    # keepdim=True so we can broadcast: (num_prompts, 1)
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)

    # Normalize: (reward - mean) / (std + eps)
    advantages = (rewards - mean_rewards) / (std_rewards + eps)

    return advantages


class GRPOLoss(nn.Module):
    """GRPO clipped surrogate objective with KL penalty.

    GRPO uses the same clipped surrogate as PPO, but with group-relative
    advantages instead of value-network advantages:

        ratio = exp(log_prob_new - log_prob_old)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1 - eps, 1 + eps) * advantages
        loss = -min(surr1, surr2) + kl_weight * KL(pi || pi_ref)

    The clipping prevents large policy updates, and the KL penalty
    keeps the policy close to the reference model.

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
            log_probs: Log probabilities from the current policy,
                       shape (batch, seq).
            old_log_probs: Log probabilities from the old policy,
                           shape (batch, seq).
            advantages: Group-relative advantages, shape (batch, seq).
                        In GRPO, these come from reward normalization
                        within each prompt's group of responses.
            log_probs_ref: Log probabilities from the reference model,
                           shape (batch, seq).
            action_mask: Optional mask for valid tokens, shape (batch, seq).

        Returns:
            Tuple of (loss, kl_divergence):
                - loss: Scalar GRPO loss (to minimize).
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
        policy_loss = -torch.min(surr1, surr2)

        # Step 5: Add KL penalty
        loss = policy_loss + self.kl_weight * kl

        # Step 6: Average over valid tokens and batch
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        kl_mean = masked_mean(kl, action_mask, dim=-1).mean()

        return loss, kl_mean


class GRPOTrainer:
    """GRPO Trainer for language model alignment.

    Orchestrates the GRPO training loop:
    1. For each prompt, generate G responses
    2. Score each response with a reward function
    3. Compute group-relative advantages (normalize within group)
    4. Update policy using clipped surrogate loss with KL penalty

    Unlike PPO, GRPO does not need:
    - A value network (advantages come from reward normalization)
    - GAE (advantages are computed directly from rewards)

    This makes GRPO simpler and often more stable than PPO.

    Args:
        model: The policy model (being trained).
        ref_model: The reference model (frozen, for KL penalty).
        reward_fn: Function that scores responses. Takes a list of
                   response strings and returns a tensor of rewards.
        clip_eps: Clipping parameter for the surrogate loss.
        kl_weight: Weight for the KL penalty.
        lr: Learning rate for the optimizer.
        max_grad_norm: Max gradient norm for gradient clipping.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[list], torch.Tensor],
        clip_eps: float = 0.2,
        kl_weight: float = 0.1,
        lr: float = 1e-5,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.max_grad_norm = max_grad_norm

        self.grpo_loss = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities.

        Args:
            model: The model to compute log probs with.
            input_ids: Token IDs, shape (batch, seq).

        Returns:
            Log probabilities of each token given previous tokens,
            shape (batch, seq-1). Position t gives log_prob of token t+1.
        """
        if model is self.ref_model:
            model.eval()

        logits = model(input_ids)  # (batch, seq, vocab)
        log_probs_all = F.log_softmax(logits, dim=-1)

        # For each position t, gather log_prob of token at position t+1
        # This gives us log P(token_{t+1} | token_{0:t})
        tokens = input_ids[:, 1:].unsqueeze(-1)  # (batch, seq-1, 1)
        log_probs = log_probs_all[:, :-1, :].gather(-1, tokens).squeeze(-1)
        return log_probs

    def step(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int,
        vocab_size: int,
        temperature: float = 1.0,
        num_responses_per_prompt: int = 4,
        tokenizer=None,
    ) -> Dict[str, float]:
        """Perform one GRPO training step.

        For each prompt:
        1. Generate G responses
        2. Score with reward function
        3. Compute group advantages
        4. Update policy with GRPO loss

        Args:
            prompts: Prompt token IDs, shape (num_prompts, prompt_len).
            max_new_tokens: Number of tokens to generate per response.
            vocab_size: Vocabulary size for generation.
            temperature: Sampling temperature for generation.
            num_responses_per_prompt: Number of responses per prompt (G).
            tokenizer: Optional tokenizer for decoding responses for reward_fn.

        Returns:
            Dict with training metrics ("loss", "kl").
        """
        device = prompts.device
        num_prompts = prompts.shape[0]
        G = num_responses_per_prompt

        # Step 1: Generate G responses for each prompt
        all_sequences = []   # (num_prompts * G, total_len)
        all_rewards = []     # (num_prompts * G,)

        self.model.eval()
        with torch.no_grad():
            for i in range(num_prompts):
                prompt = prompts[i : i + 1]  # (1, prompt_len)

                # Generate G responses
                responses = []
                for _ in range(G):
                    seq = prompt.clone()
                    for _ in range(max_new_tokens):
                        logits = self.model(seq)
                        next_logits = logits[:, -1, :] / temperature
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        seq = torch.cat([seq, next_token], dim=1)
                    responses.append(seq)

                # Stack: (G, total_len)
                group_sequences = torch.cat(responses, dim=0)
                all_sequences.append(group_sequences)

                # Score with reward function
                if tokenizer is not None:
                    response_texts = tokenizer.batch_decode(
                        group_sequences[:, prompts.shape[1]:],
                        skip_special_tokens=True,
                    )
                else:
                    # If no tokenizer, use sequences directly as "responses"
                    response_texts = [
                        group_sequences[j] for j in range(G)
                    ]

                group_rewards = self.reward_fn(response_texts)
                all_rewards.append(group_rewards)

        # Stack all sequences and rewards
        # all_sequences: (num_prompts, G, total_len) -> (num_prompts * G, total_len)
        all_sequences = torch.cat(all_sequences, dim=0)
        # all_rewards: (num_prompts, G)
        all_rewards = torch.stack(all_rewards, dim=0)

        # Step 2: Compute group advantages
        advantages = compute_group_advantages(all_rewards)  # (num_prompts, G)

        # Expand advantages to match sequence batch dimension
        # (num_prompts, G) -> (num_prompts * G,)
        advantages_flat = advantages.reshape(-1)

        # Step 3: Compute log probs (old policy, current policy, reference)
        with torch.no_grad():
            old_log_probs = self._compute_log_probs(
                self.model, all_sequences
            )
            ref_log_probs = self._compute_log_probs(
                self.ref_model, all_sequences
            )

        # Step 4: GRPO update
        self.model.train()
        self.optimizer.zero_grad()

        new_log_probs = self._compute_log_probs(self.model, all_sequences)

        # Expand advantages to match log_probs shape: (batch, seq-1)
        # advantages_flat is (num_prompts * G,), we need (num_prompts * G, seq-1)
        advantages_expanded = advantages_flat.unsqueeze(1).expand_as(new_log_probs)

        # Compute GRPO loss
        loss, kl = self.grpo_loss(
            log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages_expanded,
            log_probs_ref=ref_log_probs,
        )

        # Backward and optimize
        loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "kl": kl.item(),
        }


def grpo_step(
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

    This is a convenience function that encapsulates one gradient update
    using the GRPO loss. It handles:
    1. Computing new log probs from the current policy
    2. Computing the GRPO clipped loss with KL penalty
    3. Backward pass and optimizer step
    4. Gradient clipping

    Args:
        model: The policy model (being trained).
        ref_model: The reference model (frozen, for KL penalty).
        sequences: Full sequences (prompt + response), shape (batch, seq).
        old_log_probs: Log probs from the old policy, shape (batch, seq-1).
        ref_log_probs: Log probs from the reference model, shape (batch, seq-1).
        advantages: Group-relative advantages, shape (batch,) or (batch, seq-1).
        optimizer: Optimizer for the policy model.
        clip_eps: Clipping parameter for the surrogate loss.
        kl_weight: Weight for the KL penalty.
        max_grad_norm: Max gradient norm for gradient clipping (0 = no clipping).
        action_mask: Optional mask for valid tokens, shape (batch, seq-1).

    Returns:
        Dict with training metrics ("loss", "kl").
    """
    model.train()
    optimizer.zero_grad()

    # Compute current policy log probs
    logits = model(sequences)
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Gather log probs of the tokens that were actually generated
    tokens = sequences[:, 1:].unsqueeze(-1)  # (batch, seq-1, 1)
    new_log_probs = log_probs_all[:, :-1, :].gather(-1, tokens).squeeze(-1)

    # If advantages is 1D (per-response), expand to match log_probs shape
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1).expand_as(new_log_probs)

    # Compute GRPO loss
    grpo_loss = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
    loss, kl = grpo_loss(
        log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        log_probs_ref=ref_log_probs,
        action_mask=action_mask,
    )

    # Backward and optimize
    loss.backward()

    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return {
        "loss": loss.item(),
        "kl": kl.item(),
    }
