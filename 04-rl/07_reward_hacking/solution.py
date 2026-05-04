"""Reward Hacking -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Tuple


class RewardHackingDetectorSolution:
    """Detects when a policy is exploiting the reward model."""

    def __init__(
        self,
        reward_threshold: float = 2.0,
        diversity_threshold: float = 0.3,
        length_penalty_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.reward_threshold = reward_threshold
        self.diversity_threshold = diversity_threshold
        self.length_penalty_range = length_penalty_range
        self.reward_history: List[float] = []
        self.baseline_rewards: Optional[torch.Tensor] = None

    def set_baseline(self, rewards: torch.Tensor) -> None:
        """Set baseline reward distribution for comparison."""
        self.baseline_rewards = rewards.detach()

    def compute_reward_divergence(
        self, current_rewards: torch.Tensor
    ) -> float:
        """Measure how far current rewards have drifted from baseline."""
        if self.baseline_rewards is None:
            return 0.0

        baseline_mean = self.baseline_rewards.mean().item()
        baseline_std = self.baseline_rewards.std().item()
        current_mean = current_rewards.mean().item()

        if baseline_std < 1e-8:
            baseline_std = 1e-8

        z_score = abs(current_mean - baseline_mean) / baseline_std
        return z_score

    def compute_output_diversity(
        self, token_ids: torch.Tensor
    ) -> float:
        """Measure output diversity via unique-token ratio."""
        if token_ids.numel() == 0:
            return 0.0

        flat = token_ids.reshape(-1)
        num_unique = torch.unique(flat).numel()
        total = flat.numel()

        return num_unique / total

    def compute_length_anomaly(
        self, token_ids: torch.Tensor, expected_length: float
    ) -> float:
        """Detect abnormal output lengths."""
        if expected_length <= 0:
            return 0.0

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
        """Run all detection heuristics."""
        self.reward_history.extend(current_rewards.detach().cpu().tolist())

        divergence = self.compute_reward_divergence(current_rewards)
        diversity = self.compute_output_diversity(token_ids)
        length_anom = self.compute_length_anomaly(token_ids, expected_length)

        is_hacking = False
        if divergence > self.reward_threshold:
            is_hacking = True
        if diversity < self.diversity_threshold:
            is_hacking = True
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


class KLConstrainedRewardSolution(nn.Module):
    """Wraps a reward model and adds KL penalty from reference policy."""

    def __init__(self, reward_model: nn.Module, beta: float = 0.1):
        super().__init__()
        self.reward_model = reward_model
        self.beta = beta

    def compute_kl_penalty(
        self,
        policy_logits: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL(pi || pi_ref) per-token and average."""
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)

        kl_per_token = (reference_probs * (reference_log_probs - policy_log_probs)).sum(dim=-1)
        return kl_per_token.mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        policy_logits: Optional[torch.Tensor] = None,
        reference_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute KL-constrained reward."""
        base_reward = self.reward_model(input_ids)

        kl_penalty = torch.tensor(0.0, device=input_ids.device)
        if policy_logits is not None and reference_logits is not None:
            kl_penalty = self.compute_kl_penalty(policy_logits, reference_logits)

        effective_reward = base_reward - self.beta * kl_penalty

        return {
            "reward": base_reward,
            "kl_penalty": kl_penalty,
            "effective_reward": effective_reward,
        }


class RewardEnsembleSolution(nn.Module):
    """Averages multiple reward models for robustness."""

    def __init__(self, reward_models: List[nn.Module]):
        super().__init__()
        self.reward_models = nn.ModuleList(reward_models)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute ensemble reward scores."""
        all_rewards = []
        for rm in self.reward_models:
            rewards = rm(input_ids)
            all_rewards.append(rewards)

        stacked = torch.stack(all_rewards, dim=0)
        mean_reward = stacked.mean(dim=0)
        std_reward = stacked.std(dim=0, unbiased=False)

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "all_rewards": stacked,
        }

    def score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return just the mean reward."""
        result = self.forward(input_ids)
        return result["mean_reward"]


def analyze_reward_hacking_solution(
    reward_model: nn.Module,
    quality_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
) -> Dict[str, float]:
    """Compare reward model scores vs actual quality."""
    reward_model.eval()

    with torch.no_grad():
        rewards = reward_model(input_ids)
        qualities = quality_fn(input_ids)

    rewards = rewards.reshape(-1)
    qualities = qualities.reshape(-1)

    reward_mean = rewards.mean()
    quality_mean = qualities.mean()

    rewards_centered = rewards - reward_mean
    qualities_centered = qualities - quality_mean

    cov = (rewards_centered * qualities_centered).mean()
    reward_std = rewards_centered.std().clamp(min=1e-8)
    quality_std = qualities_centered.std().clamp(min=1e-8)

    correlation = (cov / (reward_std * quality_std)).item()

    reward_normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
    quality_normalized = (qualities - qualities.min()) / (qualities.max() - qualities.min() + 1e-8)

    inflation = (reward_normalized - quality_normalized).mean().item()
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
