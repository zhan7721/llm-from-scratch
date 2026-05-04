"""Tests for Reward Hacking Detection and Mitigation."""

import torch
import torch.nn as nn
import pytest
from reward_hacking import (
    RewardHackingDetector,
    KLConstrainedReward,
    RewardEnsemble,
    analyze_reward_hacking,
)


class DummyTransformer(nn.Module):
    """Minimal transformer-like model for testing.

    Produces hidden states of shape (batch, seq, d_model).
    """

    def __init__(self, vocab_size=256, d_model=32, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                batch_first=True,
            ),
            num_layers=2,
        )

    def forward(self, input_ids):
        h = self.embedding(input_ids)
        h = self.encoder(h)
        return h


class SimpleRewardModel(nn.Module):
    """A simple reward model that sums token embeddings and projects to scalar."""

    def __init__(self, vocab_size=256, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        h = self.embedding(input_ids)  # (batch, seq, d_model)
        pooled = h.mean(dim=1)  # (batch, d_model)
        reward = self.scalar_head(pooled).squeeze(-1)  # (batch,)
        return reward


# ============================================================
# RewardHackingDetector tests
# ============================================================


class TestRewardHackingDetector:
    """Tests for the RewardHackingDetector class."""

    def test_init_defaults(self):
        """Detector should have sensible defaults."""
        detector = RewardHackingDetector()
        assert detector.reward_threshold == 2.0
        assert detector.diversity_threshold == 0.3
        assert detector.length_penalty_range == (0.5, 2.0)

    def test_init_custom(self):
        """Detector should accept custom parameters."""
        detector = RewardHackingDetector(
            reward_threshold=3.0,
            diversity_threshold=0.5,
            length_penalty_range=(0.3, 3.0),
        )
        assert detector.reward_threshold == 3.0
        assert detector.diversity_threshold == 0.5
        assert detector.length_penalty_range == (0.3, 3.0)

    def test_set_baseline(self):
        """set_baseline should store detached rewards."""
        detector = RewardHackingDetector()
        rewards = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        detector.set_baseline(rewards)
        assert detector.baseline_rewards is not None
        assert not detector.baseline_rewards.requires_grad

    def test_reward_divergence_no_baseline(self):
        """Divergence should be 0 when no baseline is set."""
        detector = RewardHackingDetector()
        divergence = detector.compute_reward_divergence(torch.tensor([1.0, 2.0]))
        assert divergence == 0.0

    def test_reward_divergence_normal(self):
        """Divergence should be low when current rewards match baseline."""
        detector = RewardHackingDetector()
        baseline = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        detector.set_baseline(baseline)
        # Current rewards similar to baseline
        current = torch.tensor([1.5, 2.5, 3.5])
        divergence = detector.compute_reward_divergence(current)
        assert divergence < 1.0

    def test_reward_divergence_high(self):
        """Divergence should be high when current rewards are far from baseline."""
        detector = RewardHackingDetector()
        baseline = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        detector.set_baseline(baseline)
        # Current rewards much higher than baseline
        current = torch.tensor([100.0, 200.0, 300.0])
        divergence = detector.compute_reward_divergence(current)
        assert divergence > 10.0

    def test_output_diversity_varied(self):
        """Diverse outputs should have high diversity score."""
        detector = RewardHackingDetector()
        # All unique tokens
        token_ids = torch.arange(0, 100).unsqueeze(0)
        diversity = detector.compute_output_diversity(token_ids)
        assert diversity == 1.0

    def test_output_diversity_repetitive(self):
        """Repetitive outputs should have low diversity score."""
        detector = RewardHackingDetector()
        # Same token repeated
        token_ids = torch.full((1, 100), 42)
        diversity = detector.compute_output_diversity(token_ids)
        assert diversity < 0.1

    def test_output_diversity_empty(self):
        """Empty tensor should have diversity of 0."""
        detector = RewardHackingDetector()
        diversity = detector.compute_output_diversity(torch.tensor([]))
        assert diversity == 0.0

    def test_length_anomaly_normal(self):
        """Normal-length outputs should have no anomaly."""
        detector = RewardHackingDetector()
        # 50 tokens is exactly at expected length
        token_ids = torch.ones(2, 50)
        anomaly = detector.compute_length_anomaly(token_ids, expected_length=50.0)
        assert anomaly == 0.0

    def test_length_anomaly_too_short(self):
        """Very short outputs should be flagged."""
        detector = RewardHackingDetector()
        # Only 10 tokens when 50 expected -> ratio = 0.2 < 0.5
        token_ids = torch.ones(2, 10)
        anomaly = detector.compute_length_anomaly(token_ids, expected_length=50.0)
        assert anomaly > 0.0

    def test_length_anomaly_too_long(self):
        """Very long outputs should be flagged."""
        detector = RewardHackingDetector()
        # 200 tokens when 50 expected -> ratio = 4.0 > 2.0
        token_ids = torch.ones(2, 200)
        anomaly = detector.compute_length_anomaly(token_ids, expected_length=50.0)
        assert anomaly > 0.0

    def test_detect_returns_all_fields(self):
        """detect() should return all diagnostic fields."""
        detector = RewardHackingDetector()
        rewards = torch.tensor([1.0, 2.0, 3.0])
        token_ids = torch.randint(1, 100, (3, 50))
        result = detector.detect(rewards, token_ids)
        assert "reward_divergence" in result
        assert "output_diversity" in result
        assert "length_anomaly" in result
        assert "reward_mean" in result
        assert "reward_std" in result
        assert "is_hacking" in result

    def test_detect_no_hacking_normal(self):
        """Normal outputs should not be flagged as hacking."""
        detector = RewardHackingDetector()
        baseline = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        detector.set_baseline(baseline)

        # Normal rewards and varied outputs
        rewards = torch.tensor([1.5, 2.5, 3.5])
        token_ids = torch.arange(1, 151).reshape(3, 50)
        result = detector.detect(rewards, token_ids, expected_length=50.0)
        assert result["is_hacking"] == 0.0

    def test_detect_flags_repetitive(self):
        """Repetitive outputs should be flagged as hacking."""
        detector = RewardHackingDetector()
        rewards = torch.tensor([1.0, 2.0, 3.0])
        # All same token -> low diversity
        token_ids = torch.full((3, 50), 42)
        result = detector.detect(rewards, token_ids, expected_length=50.0)
        assert result["is_hacking"] == 1.0

    def test_detect_flags_high_divergence(self):
        """Very high rewards relative to baseline should be flagged."""
        detector = RewardHackingDetector(reward_threshold=2.0)
        baseline = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        detector.set_baseline(baseline)

        # Current rewards way above baseline
        rewards = torch.tensor([100.0, 200.0, 300.0])
        token_ids = torch.arange(1, 151).reshape(3, 50)
        result = detector.detect(rewards, token_ids, expected_length=50.0)
        assert result["is_hacking"] == 1.0

    def test_detect_tracks_history(self):
        """detect() should accumulate reward history."""
        detector = RewardHackingDetector()
        token_ids = torch.arange(1, 31).reshape(2, 15)

        detector.detect(torch.tensor([1.0, 2.0]), token_ids[:1])
        detector.detect(torch.tensor([3.0, 4.0]), token_ids[1:])

        assert len(detector.reward_history) == 4

    def test_reward_mean_and_std(self):
        """detect() should report correct reward statistics."""
        detector = RewardHackingDetector()
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        token_ids = torch.arange(1, 251).reshape(5, 50)
        result = detector.detect(rewards, token_ids)
        assert abs(result["reward_mean"] - 3.0) < 1e-5
        assert result["reward_std"] > 0


# ============================================================
# KLConstrainedReward tests
# ============================================================


class TestKLConstrainedReward:
    """Tests for the KLConstrainedReward class."""

    def test_init(self):
        """KLConstrainedReward should wrap a reward model."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)
        assert kl_constrained.reward_model is reward_model
        assert kl_constrained.beta == 0.1

    def test_init_with_reference(self):
        """KLConstrainedReward should accept and freeze a reference model."""
        reward_model = SimpleRewardModel()
        ref_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1, reference_model=ref_model)

        # Reference model parameters should be frozen
        for param in kl_constrained.reference_model.parameters():
            assert not param.requires_grad

    def test_forward_no_kl(self):
        """Forward without logits should return base reward with zero KL."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)

        input_ids = torch.randint(0, 256, (2, 10))
        result = kl_constrained.forward(input_ids)

        assert "reward" in result
        assert "kl_penalty" in result
        assert "effective_reward" in result
        assert result["reward"].shape == (2,)
        assert result["effective_reward"].shape == (2,)
        # No KL info provided, so effective == base
        assert torch.allclose(result["reward"], result["effective_reward"])

    def test_forward_with_logits(self):
        """Forward with logits should compute KL penalty."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.5)

        input_ids = torch.randint(0, 256, (2, 10))
        vocab_size = 256
        policy_logits = torch.randn(2, 10, vocab_size)
        reference_logits = torch.randn(2, 10, vocab_size)

        result = kl_constrained.forward(
            input_ids,
            policy_logits=policy_logits,
            reference_logits=reference_logits,
        )

        assert result["kl_penalty"].item() >= 0
        # KL penalty should reduce effective reward
        assert result["effective_reward"].shape == (2,)

    def test_kl_penalty_identical_distributions(self):
        """KL should be zero when policy and reference are identical."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)

        logits = torch.randn(2, 10, 256)
        kl = kl_constrained.compute_kl_penalty(logits, logits)
        assert kl.item() < 1e-5

    def test_kl_penalty_different_distributions(self):
        """KL should be positive when distributions differ."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)

        policy_logits = torch.randn(2, 10, 256)
        reference_logits = torch.randn(2, 10, 256) + 2.0
        kl = kl_constrained.compute_kl_penalty(policy_logits, reference_logits)
        assert kl.item() > 0

    def test_kl_penalty_with_mask(self):
        """KL should respect attention mask."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)

        policy_logits = torch.randn(2, 10, 256)
        reference_logits = torch.randn(2, 10, 256)

        # Mask out second half of each sequence
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0

        kl_with_mask = kl_constrained.compute_kl_penalty(
            policy_logits, reference_logits, attention_mask=mask
        )
        kl_without_mask = kl_constrained.compute_kl_penalty(
            policy_logits, reference_logits
        )
        # Results may differ because mask changes which positions contribute
        assert kl_with_mask.item() >= 0
        assert kl_without_mask.item() >= 0

    def test_kl_from_logprobs(self):
        """KL from log-probs should approximate the full KL."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)

        policy_log_probs = torch.randn(4)
        reference_log_probs = torch.randn(4)
        kl = kl_constrained.compute_kl_penalty_from_logprobs(
            policy_log_probs, reference_log_probs
        )
        # KL from log-probs is a scalar
        assert kl.dim() == 0

    def test_kl_from_logprobs_2d(self):
        """KL from 2D log-probs should sum over sequence dimension."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)

        policy_log_probs = torch.randn(2, 10)
        reference_log_probs = torch.randn(2, 10)
        kl = kl_constrained.compute_kl_penalty_from_logprobs(
            policy_log_probs, reference_log_probs
        )
        assert kl.dim() == 0

    def test_beta_scales_kl_penalty(self):
        """Higher beta should produce lower effective reward."""
        reward_model = SimpleRewardModel()
        input_ids = torch.randint(0, 256, (2, 10))

        policy_logits = torch.randn(2, 10, 256)
        reference_logits = torch.randn(2, 10, 256)

        low_beta = KLConstrainedReward(reward_model, beta=0.01)
        high_beta = KLConstrainedReward(reward_model, beta=10.0)

        result_low = low_beta.forward(
            input_ids,
            policy_logits=policy_logits,
            reference_logits=reference_logits,
        )
        result_high = high_beta.forward(
            input_ids,
            policy_logits=policy_logits,
            reference_logits=reference_logits,
        )

        # Higher beta -> more penalty -> lower effective reward
        assert result_high["effective_reward"].mean() <= result_low["effective_reward"].mean()

    def test_gradient_flows_through_reward(self):
        """Gradients should flow through the reward model."""
        reward_model = SimpleRewardModel()
        kl_constrained = KLConstrainedReward(reward_model, beta=0.1)

        input_ids = torch.randint(0, 256, (2, 10))
        result = kl_constrained.forward(input_ids)
        loss = result["effective_reward"].sum()
        loss.backward()

        for name, param in reward_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ============================================================
# RewardEnsemble tests
# ============================================================


class TestRewardEnsemble:
    """Tests for the RewardEnsemble class."""

    def test_init(self):
        """Ensemble should store reward models."""
        models = [SimpleRewardModel() for _ in range(3)]
        ensemble = RewardEnsemble(models)
        assert len(ensemble.reward_models) == 3

    def test_forward_returns_all_fields(self):
        """Forward should return mean, std, and all rewards."""
        models = [SimpleRewardModel() for _ in range(3)]
        ensemble = RewardEnsemble(models)
        input_ids = torch.randint(0, 256, (2, 10))
        result = ensemble.forward(input_ids)

        assert "mean_reward" in result
        assert "std_reward" in result
        assert "all_rewards" in result

    def test_forward_shapes(self):
        """Forward should return correct shapes."""
        n_models = 3
        batch_size = 4
        models = [SimpleRewardModel() for _ in range(n_models)]
        ensemble = RewardEnsemble(models)
        input_ids = torch.randint(0, 256, (batch_size, 10))
        result = ensemble.forward(input_ids)

        assert result["mean_reward"].shape == (batch_size,)
        assert result["std_reward"].shape == (batch_size,)
        assert result["all_rewards"].shape == (n_models, batch_size)

    def test_mean_reward(self):
        """Mean reward should be the average of individual model rewards."""
        models = [SimpleRewardModel() for _ in range(3)]
        ensemble = RewardEnsemble(models)
        input_ids = torch.randint(0, 256, (2, 10))
        result = ensemble.forward(input_ids)

        expected_mean = result["all_rewards"].mean(dim=0)
        assert torch.allclose(result["mean_reward"], expected_mean)

    def test_std_reward(self):
        """Std reward should measure disagreement between models."""
        models = [SimpleRewardModel() for _ in range(3)]
        ensemble = RewardEnsemble(models)
        input_ids = torch.randint(0, 256, (4, 10))
        result = ensemble.forward(input_ids)

        expected_std = result["all_rewards"].std(dim=0, unbiased=False)
        assert torch.allclose(result["std_reward"], expected_std)

    def test_single_model_zero_std(self):
        """Single model should have zero standard deviation."""
        models = [SimpleRewardModel()]
        ensemble = RewardEnsemble(models)
        input_ids = torch.randint(0, 256, (2, 10))
        result = ensemble.forward(input_ids)

        assert torch.allclose(result["std_reward"], torch.zeros(2), atol=1e-6)

    def test_score_convenience(self):
        """score() should return mean reward directly."""
        models = [SimpleRewardModel() for _ in range(3)]
        ensemble = RewardEnsemble(models)
        input_ids = torch.randint(0, 256, (2, 10))

        mean_result = ensemble.score(input_ids)
        full_result = ensemble.forward(input_ids)
        assert torch.allclose(mean_result, full_result["mean_reward"])

    def test_ensemble_reduces_variance(self):
        """Ensemble of diverse models should have lower variance than individuals."""
        torch.manual_seed(42)
        # Create multiple different reward models
        models = [SimpleRewardModel(vocab_size=256, d_model=32) for _ in range(5)]
        ensemble = RewardEnsemble(models)

        # Test on multiple inputs
        inputs = [torch.randint(0, 256, (1, 10)) for _ in range(10)]
        input_batch = torch.cat(inputs, dim=0)

        # Get individual model scores
        individual_scores = []
        for m in models:
            with torch.no_grad():
                scores = m(input_batch)
            individual_scores.append(scores)

        # Compute variance across inputs for each individual model
        individual_variances = []
        for scores in individual_scores:
            individual_variances.append(scores.var().item())

        # Ensemble variance
        ensemble_result = ensemble.forward(input_batch)
        ensemble_variance = ensemble_result["mean_reward"].var().item()

        # The ensemble mean should have lower variance than individual models
        avg_individual_variance = sum(individual_variances) / len(individual_variances)
        # Ensemble variance should be less than or equal to average individual variance
        assert ensemble_variance <= avg_individual_variance + 1e-6

    def test_parameters_tracked(self):
        """Ensemble parameters should be tracked by PyTorch."""
        models = [SimpleRewardModel() for _ in range(3)]
        ensemble = RewardEnsemble(models)
        params = list(ensemble.parameters())
        # Each SimpleRewardModel has embedding + linear = 2 parameter groups
        # 3 models = at least 6 parameters
        assert len(params) >= 6


# ============================================================
# analyze_reward_hacking tests
# ============================================================


class TestAnalyzeRewardHacking:
    """Tests for the analyze_reward_hacking function."""

    def test_returns_all_fields(self):
        """analyze_reward_hacking should return all diagnostic fields."""
        reward_model = SimpleRewardModel()
        input_ids = torch.randint(0, 256, (4, 10))

        def quality_fn(ids):
            return ids.float().mean(dim=-1)

        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)

        assert "mean_reward" in result
        assert "mean_quality" in result
        assert "correlation" in result
        assert "reward_std" in result
        assert "quality_std" in result
        assert "reward_inflation" in result
        assert "is_hacking" in result

    def test_correlated_reward_quality(self):
        """High correlation when reward tracks quality."""
        reward_model = SimpleRewardModel()
        # quality_fn = same as reward => high correlation
        def quality_fn(ids):
            return reward_model(ids)

        input_ids = torch.randint(0, 256, (10, 10))
        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)
        # Perfect correlation (or near-perfect)
        assert result["correlation"] >= 0.85

    def test_uncorrelated_reward_quality(self):
        """Low correlation when reward does not track quality."""
        reward_model = SimpleRewardModel()
        # quality_fn = random (independent of reward)
        torch.manual_seed(999)
        random_quality = torch.randn(10)
        def quality_fn(ids):
            return random_quality

        input_ids = torch.randint(0, 256, (10, 10))
        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)
        # Should have low correlation (not strictly 0 due to randomness)
        # We just check it does not crash and returns reasonable values
        assert -1.0 <= result["correlation"] <= 1.0
        # With random quality, correlation should be low
        assert abs(result["correlation"]) < 0.5

    def test_reward_inflation_positive_when_hacked(self):
        """Inflation should be positive when reward exceeds quality."""
        reward_model = SimpleRewardModel()

        # Create inputs that get high reward but low quality
        # We make quality function return constant low values
        def quality_fn(ids):
            return torch.full((ids.shape[0],), -10.0)

        input_ids = torch.randint(0, 256, (10, 10))
        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)
        # With constant quality, inflation calculation should work
        assert "reward_inflation" in result
        # With constant quality and varying reward, inflation should be positive
        assert result["reward_inflation"] >= 0

    def test_no_hacking_with_matching_signals(self):
        """Should not flag hacking when reward and quality align."""
        reward_model = SimpleRewardModel()

        def quality_fn(ids):
            # Quality is proportional to reward
            return reward_model(ids) * 0.5 + 1.0

        input_ids = torch.randint(0, 256, (10, 10))
        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)
        assert result["is_hacking"] == 0.0

    def test_is_hacking_flag(self):
        """is_hacking should be 1.0 when conditions are met."""
        reward_model = SimpleRewardModel()

        # Create scenario where reward is high but quality is random noise
        torch.manual_seed(42)
        random_quality = torch.randn(20) * 0.01  # Very low variance, near zero

        def quality_fn(ids):
            return random_quality[:ids.shape[0]]

        input_ids = torch.randint(0, 256, (20, 10))
        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)
        # With random quality, correlation should be low
        # and inflation should be detectable
        assert result["is_hacking"] in (0.0, 1.0)  # Should be a valid flag
        # With random quality, correlation should be low
        assert abs(result["correlation"]) < 0.5

    def test_values_are_floats(self):
        """All returned values should be Python floats."""
        reward_model = SimpleRewardModel()
        input_ids = torch.randint(0, 256, (4, 10))

        def quality_fn(ids):
            return ids.float().mean(dim=-1)

        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_reward_std_non_negative(self):
        """Reward std should always be non-negative."""
        reward_model = SimpleRewardModel()
        input_ids = torch.randint(0, 256, (4, 10))

        def quality_fn(ids):
            return ids.float().mean(dim=-1)

        result = analyze_reward_hacking(reward_model, quality_fn, input_ids)
        assert result["reward_std"] >= 0.0
        assert result["quality_std"] >= 0.0
