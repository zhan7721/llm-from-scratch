"""Reward Hacking Detection and Mitigation.

This module implements tools to detect and prevent reward hacking (also called
reward overoptimization) -- the phenomenon where a policy exploits imperfections
in the reward model to achieve high reward scores without actually improving
output quality.

Key concepts:
- RewardHackingDetector: flags suspicious patterns like degenerate outputs
- KLConstrainedReward: adds KL penalty to keep policy near reference
- RewardEnsemble: averages multiple reward models for robustness
- analyze_reward_hacking: diagnostic comparing reward vs actual quality

Reward hacking is one of the central challenges in RLHF. As the policy is
optimized against the reward model, it may find "shortcuts" that maximize
the reward signal without corresponding to genuine quality improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Tuple




# ============================================================
# RewardHackingDetector
# ============================================================


class RewardHackingDetector:
    """Detects when a policy is exploiting the reward model.

    Monitors several diagnostic signals that indicate reward hacking:
    - Reward divergence: how far rewards have drifted from a reference
    - Output diversity: whether outputs have become repetitive
    - Length anomalies: whether output lengths are abnormal
    - Reward inflation: high rewards paired with low actual quality

    These signals can be used to trigger early stopping or adjust
    the KL penalty coefficient during RL training.

    Args:
        reward_threshold: Z-score threshold for flagging anomalous rewards.
        diversity_threshold: Minimum unique-token ratio (0-1) before flagging.
        length_penalty_range: (min_ratio, max_ratio) of expected output length.
    """

    def __init__(
        self,
        reward_threshold: float = 2.0,
        diversity_threshold: float = 0.3,
        length_penalty_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.reward_threshold = reward_threshold
        self.diversity_threshold = diversity_threshold
        self.length_penalty_range = length_penalty_range

        # History tracking
        self.reward_history: List[float] = []
        self.baseline_rewards: Optional[torch.Tensor] = None

    def set_baseline(self, rewards: torch.Tensor) -> None:
        """Set baseline reward distribution for comparison.

        Args:
            rewards: Tensor of reward scores from a reference policy.
        """
        self.baseline_rewards = rewards.detach()

    def compute_reward_divergence(
        self, current_rewards: torch.Tensor
    ) -> float:
        """Measure how far current rewards have drifted from baseline.

        Uses a z-score to compare the current mean reward to the
        baseline distribution. Higher z-score = more divergence.

        Args:
            current_rewards: Current batch of reward scores.

        Returns:
            Scalar divergence score (higher = more diverged).
        """
        if self.baseline_rewards is None:
            return 0.0

        # Compare means relative to baseline std
        baseline_mean = self.baseline_rewards.mean().item()
        baseline_std = self.baseline_rewards.std().item()
        current_mean = current_rewards.mean().item()

        # Avoid division by zero
        if baseline_std < 1e-8:
            baseline_std = 1e-8

        # Z-score of current mean relative to baseline
        z_score = abs(current_mean - baseline_mean) / baseline_std
        return z_score

    def compute_output_diversity(
        self, token_ids: torch.Tensor
    ) -> float:
        """Measure output diversity via unique-token ratio.

        Repetitive outputs are a hallmark of reward hacking -- the model
        may have found a "template" that scores well but lacks variety.

        Args:
            token_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Diversity score in [0, 1]. Higher = more diverse.
        """
        if token_ids.numel() == 0:
            return 0.0

        # Flatten and count unique tokens
        flat = token_ids.reshape(-1)
        num_unique = torch.unique(flat).numel()
        total = flat.numel()

        return num_unique / total

    def compute_length_anomaly(
        self, token_ids: torch.Tensor, expected_length: float
    ) -> float:
        """Detect abnormal output lengths.

        Reward hacking often manifests as outputs that are either
        very short (degenerate) or very long (padding/rambling).

        Args:
            token_ids: Token IDs of shape (batch, seq_len).
            expected_length: Expected average sequence length.

        Returns:
            Anomaly score. 0 = normal, >0 = increasingly anomalous.
        """
        if expected_length <= 0:
            return 0.0

        # Compute actual lengths (non-zero tokens, assuming 0 is pad)
        lengths = (token_ids != 0).sum(dim=-1).float()
        avg_length = lengths.mean().item()

        ratio = avg_length / expected_length
        min_ratio, max_ratio = self.length_penalty_range

        if ratio < min_ratio:
            return (min_ratio - ratio) / min_ratio
        elif ratio > max_ratio:
            return (ratio - max_ratio) / max_ratio
        else:
            return 0.0

    def detect(
        self,
        current_rewards: torch.Tensor,
        token_ids: torch.Tensor,
        expected_length: float = 50.0,
    ) -> Dict[str, float]:
        """Run all detection heuristics and return diagnostic signals.

        Args:
            current_rewards: Current batch of reward scores.
            token_ids: Generated token IDs of shape (batch, seq_len).
            expected_length: Expected average output length.

        Returns:
            Dict of diagnostic signals:
            - "reward_divergence": how far rewards drifted from baseline
            - "output_diversity": unique-token ratio (lower = more suspicious)
            - "length_anomaly": how abnormal the output lengths are
            - "reward_mean": current mean reward
            - "reward_std": current reward standard deviation
            - "is_hacking": 1.0 if any signal exceeds threshold, else 0.0
        """
        # Track reward history
        self.reward_history.extend(current_rewards.detach().cpu().tolist())

        # Compute all signals
        divergence = self.compute_reward_divergence(current_rewards)
        diversity = self.compute_output_diversity(token_ids)
        length_anom = self.compute_length_anomaly(token_ids, expected_length)

        # Determine if hacking is likely
        is_hacking = False

        # High reward divergence
        if divergence > self.reward_threshold:
            is_hacking = True

        # Low output diversity
        if diversity < self.diversity_threshold:
            is_hacking = True

        # Abnormal lengths
        if length_anom > 0.5:
            is_hacking = True

        return {
            "reward_divergence": divergence,
            "output_diversity": diversity,
            "length_anomaly": length_anom,
            "reward_mean": current_rewards.mean().item(),
            "reward_std": current_rewards.std().item(),
            "is_hacking": float(is_hacking),
        }


# ============================================================
# KLConstrainedReward
# ============================================================


class KLConstrainedReward(nn.Module):
    """Wraps a reward model and adds KL penalty from reference policy.

    The effective reward is:
        effective_reward = reward - beta * KL(policy || reference)

    This prevents the policy from moving too far from the reference,
    which limits the policy's ability to exploit reward model weaknesses.

    The KL divergence is computed per-token as:
        KL = sum_t sum_v pi_ref(v|x_{<t}) * (log pi_ref(v|x_{<t}) - log pi(v|x_{<t}))

    For simplicity, we approximate this using log-probability differences
    between the policy and reference on the generated tokens.

    Args:
        reward_model: The base reward model to wrap.
        beta: KL penalty coefficient (higher = stronger regularization).
        reference_model: Reference policy model (frozen copy of initial policy).
    """

    def __init__(
        self,
        reward_model: nn.Module,
        beta: float = 0.1,
        reference_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.reward_model = reward_model
        self.beta = beta
        self.reference_model = reference_model

        # Freeze reference model if provided
        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False

    def compute_kl_penalty(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence between reference and policy distributions.

        KL(pi_ref || pi) = sum_v pi_ref(v) * (log pi_ref(v) - log pi(v))

        We compute this per-token and then average over positions and batch.

        Args:
            policy_logits: Logits from the policy, shape (batch, seq, vocab).
            reference_logits: Logits from the reference, shape (batch, seq, vocab).
            attention_mask: Optional mask, shape (batch, seq). 1 for real tokens.

        Returns:
            Scalar KL penalty (mean over batch).
        """
        # Convert to log-probabilities
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)

        # Reference probabilities (for the expectation)
        reference_probs = F.softmax(reference_logits, dim=-1)

        # KL = sum_v p_ref(v) * (log p_ref(v) - log p(v))
        # Shape: (batch, seq, vocab) -> sum over vocab -> (batch, seq)
        kl_per_token = (reference_probs * (reference_log_probs - policy_log_probs)).sum(dim=-1)

        if attention_mask is not None:
            # Mask out padding positions
            kl_per_token = kl_per_token * attention_mask
            # Average over non-masked positions
            kl_mean = kl_per_token.sum() / attention_mask.sum().clamp(min=1)
        else:
            kl_mean = kl_per_token.mean()

        return kl_mean

    def compute_kl_penalty_from_logprobs(
        self,
        policy_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute simplified KL penalty from sequence-level log-probabilities.

        When we only have the log-probabilities of generated tokens (not full
        logits), we approximate:
            KL ~ E[log pi(a|s) - log pi_ref(a|s)]

        This is a Monte Carlo estimate of the KL divergence.

        Args:
            policy_log_probs: Log-probs from policy, shape (batch,) or (batch, seq).
            reference_log_probs: Log-probs from reference, same shape.

        Returns:
            Scalar KL penalty.
        """
        # Simple approximation: difference of log probs
        kl = policy_log_probs - reference_log_probs

        if kl.dim() > 1:
            # Sum over sequence dimension, mean over batch
            kl = kl.sum(dim=-1).mean()
        else:
            kl = kl.mean()

        return kl

    def forward(
        self,
        input_ids: torch.Tensor,
        policy_logits: Optional[torch.Tensor] = None,
        reference_logits: Optional[torch.Tensor] = None,
        policy_log_probs: Optional[torch.Tensor] = None,
        reference_log_probs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute KL-constrained reward.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            policy_logits: Policy logits, shape (batch, seq, vocab). Optional.
            reference_logits: Reference logits, same shape. Optional.
            policy_log_probs: Policy log-probs. Optional (alternative to logits).
            reference_log_probs: Reference log-probs. Optional.
            attention_mask: Attention mask, shape (batch, seq). Optional.

        Returns:
            Dict with:
            - "reward": base reward from reward model, shape (batch,)
            - "kl_penalty": KL divergence, scalar
            - "effective_reward": reward - beta * KL, shape (batch,)
        """
        # Compute base reward
        base_reward = self.reward_model(input_ids)

        # Compute KL penalty
        kl_penalty = torch.tensor(0.0, device=input_ids.device)

        if policy_logits is not None and reference_logits is not None:
            kl_penalty = self.compute_kl_penalty(
                policy_logits, reference_logits, attention_mask
            )
        elif policy_log_probs is not None and reference_log_probs is not None:
            kl_penalty = self.compute_kl_penalty_from_logprobs(
                policy_log_probs, reference_log_probs
            )

        # Effective reward = base reward - beta * KL
        effective_reward = base_reward - self.beta * kl_penalty

        return {
            "reward": base_reward,
            "kl_penalty": kl_penalty,
            "effective_reward": effective_reward,
        }


# ============================================================
# RewardEnsemble
# ============================================================


class RewardEnsemble(nn.Module):
    """Averages multiple reward models for more robust scoring.

    Individual reward models can have blind spots or biases that the
    policy might exploit. By averaging scores from multiple models,
    we reduce the variance of the reward signal and make it harder
    for the policy to find exploits that work across all models.

    The ensemble also provides a disagreement signal (standard deviation
    across models) that indicates uncertainty -- high disagreement means
    the models are unsure, which may indicate the input is out of
    distribution or adversarial.

    Args:
        reward_models: List of reward model instances.
    """

    def __init__(self, reward_models: List[nn.Module]):
        super().__init__()
        # Register as ModuleList so parameters are tracked
        self.reward_models = nn.ModuleList(reward_models)

    def forward(
        self, input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute ensemble reward scores.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).

        Returns:
            Dict with:
            - "mean_reward": averaged reward, shape (batch,)
            - "std_reward": standard deviation (disagreement), shape (batch,)
            - "all_rewards": stacked individual rewards, shape (n_models, batch)
        """
        # Collect rewards from each model
        all_rewards = []
        for rm in self.reward_models:
            rewards = rm(input_ids)
            all_rewards.append(rewards)

        # Stack: (n_models, batch)
        stacked = torch.stack(all_rewards, dim=0)

        # Statistics across models
        mean_reward = stacked.mean(dim=0)  # (batch,)
        # Use unbiased=False to avoid NaN when n_models=1
        # (torch.std with Bessel's correction divides by n-1, which is 0 for n=1)
        std_reward = stacked.std(dim=0, unbiased=False)  # (batch,)

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "all_rewards": stacked,
        }

    def score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convenience method: return just the mean reward.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).

        Returns:
            Mean reward scores, shape (batch,).
        """
        result = self.forward(input_ids)
        return result["mean_reward"]


# ============================================================
# analyze_reward_hacking
# ============================================================


def analyze_reward_hacking(
    reward_model: nn.Module,
    quality_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
) -> Dict[str, float]:
    """Diagnostic tool comparing reward model scores vs actual quality.

    This function helps identify reward hacking by comparing what the
    reward model says about outputs vs some independent quality measure.
    If reward is high but quality is low, the reward model is being hacked.

    Args:
        reward_model: The reward model to evaluate.
        quality_fn: A function that takes token_ids and returns quality
                    scores of shape (batch,). This could measure fluency,
                    coherence, factual accuracy, etc.
        input_ids: Generated token IDs to evaluate, shape (batch, seq_len).

    Returns:
        Dict of diagnostic metrics:
        - "mean_reward": average reward model score
        - "mean_quality": average quality score
        - "correlation": Pearson correlation between reward and quality
        - "reward_std": standard deviation of reward scores
        - "quality_std": standard deviation of quality scores
        - "reward_inflation": reward - quality (positive = inflated)
        - "is_hacking": 1.0 if reward inflation is high, else 0.0
    """
    was_training = reward_model.training
    reward_model.eval()

    try:
        with torch.no_grad():
            rewards = reward_model(input_ids)
            qualities = quality_fn(input_ids)

        # Ensure 1D
        rewards = rewards.reshape(-1)
        qualities = qualities.reshape(-1)

        # Compute correlation
        reward_mean = rewards.mean()
        quality_mean = qualities.mean()

        rewards_centered = rewards - reward_mean
        qualities_centered = qualities - quality_mean

        cov = (rewards_centered * qualities_centered).mean()
        reward_std = rewards_centered.std().clamp(min=1e-8)
        quality_std = qualities_centered.std().clamp(min=1e-8)

        correlation = (cov / (reward_std * quality_std)).item()

        # Reward inflation: how much reward exceeds quality
        # Normalize both to [0, 1] for comparison
        reward_normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        quality_normalized = (qualities - qualities.min()) / (qualities.max() - qualities.min() + 1e-8)

        inflation = (reward_normalized - quality_normalized).mean().item()

        # Flag hacking if inflation is high and correlation is low
        is_hacking = inflation > 0.2 and correlation < 0.3

        return {
            "mean_reward": reward_mean.item(),
            "mean_quality": quality_mean.item(),
            "correlation": correlation,
            "reward_std": reward_std.item(),
            "quality_std": quality_std.item(),
            "reward_inflation": inflation,
            "is_hacking": float(is_hacking),
        }
    finally:
        if was_training:
            reward_model.train()
