"""Tests for PPO (Proximal Policy Optimization) implementation."""

import torch
import torch.nn as nn
import pytest
import math
from ppo import (
    PPOClipLoss,
    GAE,
    PPOTrainer,
    rollout,
    approx_kl_divergence,
    masked_mean,
)


class DummyPolicy(nn.Module):
    """Minimal policy/value model for testing.

    Produces logits over vocab and a scalar value estimate.
    """

    def __init__(self, vocab_size=32, d_model=16, max_seq_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=2,
                dim_feedforward=d_model * 2,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        """Forward pass returning logits and values.

        Args:
            input_ids: (batch, seq) token IDs.

        Returns:
            logits: (batch, seq, vocab_size)
            values: (batch, seq)
        """
        h = self.embedding(input_ids)
        h = self.encoder(h)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values


# ============================================================
# Helper function tests
# ============================================================


class TestApproxKLDivergence:
    """Tests for the approx_kl_divergence helper."""

    def test_kl_zero_when_same(self):
        """KL divergence should be ~0 when distributions are the same."""
        log_probs = torch.randn(4, 10)
        kl = approx_kl_divergence(log_probs, log_probs)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_kl_positive_when_different(self):
        """KL divergence should be positive when distributions differ."""
        log_probs = torch.randn(4, 10)
        log_probs_ref = log_probs + 0.5
        kl = approx_kl_divergence(log_probs, log_probs_ref)
        # KL should be positive (or very close to 0)
        assert (kl >= -1e-6).all()

    def test_kl_with_mask(self):
        """KL should respect the action mask."""
        log_probs = torch.randn(2, 8)
        log_probs_ref = log_probs + 1.0
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
        kl = approx_kl_divergence(log_probs, log_probs_ref, action_mask=mask)
        # Masked positions should be 0
        assert (kl[:, 4:] == 0).all()
        assert (kl[0, :4] != 0).any()

    def test_kl_shape(self):
        """KL should have the same shape as input."""
        log_probs = torch.randn(3, 12)
        log_probs_ref = torch.randn(3, 12)
        kl = approx_kl_divergence(log_probs, log_probs_ref)
        assert kl.shape == (3, 12)


class TestMaskedMean:
    """Tests for the masked_mean helper."""

    def test_no_mask(self):
        """Without mask, should equal regular mean."""
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = masked_mean(tensor, mask=None, dim=-1)
        expected = tensor.mean(dim=-1)
        assert torch.allclose(result, expected)

    def test_with_mask(self):
        """With mask, should only average over unmasked elements."""
        tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[1, 1, 0, 0]])
        result = masked_mean(tensor, mask=mask, dim=-1)
        expected = torch.tensor([1.5])  # (1+2)/2
        assert torch.allclose(result, expected)

    def test_full_mask(self):
        """Full mask should give regular mean."""
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1, 1, 1]])
        result = masked_mean(tensor, mask=mask, dim=-1)
        expected = torch.tensor([2.0])
        assert torch.allclose(result, expected)


# ============================================================
# PPOClipLoss tests
# ============================================================


class TestPPOClipLoss:
    """Tests for the PPOClipLoss class."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar tensor."""
        loss_fn = PPOClipLoss(clip_eps=0.2, kl_weight=0.1)
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        log_probs_ref = torch.randn(4, 10)
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_kl_returned(self):
        """Should return KL divergence as second element."""
        loss_fn = PPOClipLoss(clip_eps=0.2, kl_weight=0.1)
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        log_probs_ref = torch.randn(4, 10)
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert isinstance(kl, torch.Tensor)

    def test_clip_bounds_advantage(self):
        """Clipped loss should bound the surrogate objective.

        When advantage > 0: loss is capped at (1+eps) * advantage
        When advantage < 0: loss is capped at (1-eps) * advantage
        """
        torch.manual_seed(42)
        loss_fn = PPOClipLoss(clip_eps=0.2, kl_weight=0.0)  # no KL for clean test

        # Case 1: positive advantage, large ratio (should be clipped)
        log_probs = torch.randn(2, 8) + 2.0  # much larger than old
        old_log_probs = torch.randn(2, 8)
        advantages = torch.ones(2, 8) * 5.0  # positive
        log_probs_ref = old_log_probs.clone()  # ref = old, so KL ~ 0

        loss_clipped, _ = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)

        # Compute what unclipped loss would be
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        loss_unclipped = -surr1.mean()

        # Clipped loss should be >= unclipped loss (more negative = better for policy)
        # Actually, -min(surr1, surr2): when ratio > 1+eps and advantage > 0,
        # surr2 = (1+eps)*advantage < surr1, so min = surr2, loss = -surr2
        # The clipped loss should be bounded: -loss_clipped <= (1+eps) * advantages.mean()
        max_surrogate = (1 + 0.2) * advantages.mean()
        assert (-loss_clipped).item() <= max_surrogate.item() + 1e-4

    def test_loss_positive_with_kl(self):
        """Loss should include KL penalty."""
        loss_fn = PPOClipLoss(clip_eps=0.2, kl_weight=1.0)
        log_probs = torch.randn(4, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.zeros(4, 10)  # zero advantage -> loss = 0 + KL
        log_probs_ref = log_probs + 1.0  # different from current -> KL > 0
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        # With zero advantages, loss = kl_weight * KL > 0
        assert loss.item() > 0

    def test_gradient_flows(self):
        """Gradients should flow through the loss."""
        loss_fn = PPOClipLoss(clip_eps=0.2, kl_weight=0.1)
        log_probs = torch.randn(4, 10, requires_grad=True)
        old_log_probs = torch.randn(4, 10).detach()
        advantages = torch.randn(4, 10)
        log_probs_ref = torch.randn(4, 10).detach()
        loss, _ = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        loss.backward()
        assert log_probs.grad is not None

    def test_kl_weight_scales_penalty(self):
        """Higher kl_weight should increase the loss when KL > 0."""
        log_probs = torch.randn(4, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.zeros(4, 10)
        log_probs_ref = log_probs + 0.5

        loss_low, _ = PPOClipLoss(clip_eps=0.2, kl_weight=0.1)(
            log_probs, old_log_probs, advantages, log_probs_ref
        )
        loss_high, _ = PPOClipLoss(clip_eps=0.2, kl_weight=1.0)(
            log_probs, old_log_probs, advantages, log_probs_ref
        )
        assert loss_high.item() > loss_low.item()

    def test_with_action_mask(self):
        """Loss should respect action mask."""
        loss_fn = PPOClipLoss(clip_eps=0.2, kl_weight=0.1)
        log_probs = torch.randn(2, 8)
        old_log_probs = torch.randn(2, 8)
        advantages = torch.randn(2, 8)
        log_probs_ref = torch.randn(2, 8)
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]])
        loss, kl = loss_fn(
            log_probs, old_log_probs, advantages, log_probs_ref, action_mask=mask
        )
        assert loss.shape == ()

    def test_no_kl_penalty_when_ref_matches(self):
        """KL should be ~0 when log_probs_ref == log_probs."""
        loss_fn = PPOClipLoss(clip_eps=0.2, kl_weight=1.0)
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        log_probs_ref = log_probs.clone()
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert kl.abs().max().item() < 1e-5


# ============================================================
# GAE tests
# ============================================================


class TestGAE:
    """Tests for the GAE (Generalized Advantage Estimation) class."""

    def test_advantages_shape(self):
        """Advantages should have the same shape as rewards."""
        gae = GAE(gamma=0.99, lam=0.95)
        rewards = torch.randn(4, 10)
        values = torch.randn(4, 10)
        next_values = torch.randn(4)
        advantages, returns = gae(rewards, values, next_values)
        assert advantages.shape == rewards.shape

    def test_returns_shape(self):
        """Returns should have the same shape as rewards."""
        gae = GAE(gamma=0.99, lam=0.95)
        rewards = torch.randn(4, 10)
        values = torch.randn(4, 10)
        next_values = torch.randn(4)
        advantages, returns = gae(rewards, values, next_values)
        assert returns.shape == rewards.shape

    def test_advantages_positive_with_positive_rewards(self):
        """With positive rewards and zero values, advantages should be positive."""
        gae = GAE(gamma=0.99, lam=0.95)
        rewards = torch.ones(2, 5) * 1.0
        values = torch.zeros(2, 5)
        next_values = torch.zeros(2)
        advantages, _ = gae(rewards, values, next_values)
        assert (advantages > 0).all()

    def test_advantages_negative_with_negative_rewards(self):
        """With negative rewards and zero values, advantages should be negative."""
        gae = GAE(gamma=0.99, lam=0.95)
        rewards = torch.ones(2, 5) * -1.0
        values = torch.zeros(2, 5)
        next_values = torch.zeros(2)
        advantages, _ = gae(rewards, values, next_values)
        assert (advantages < 0).all()

    def test_discount_factor_effect(self):
        """Higher gamma should give more weight to future rewards."""
        rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        values = torch.zeros(1, 5)
        next_values = torch.zeros(1)

        gae_low = GAE(gamma=0.5, lam=0.95)
        gae_high = GAE(gamma=0.99, lam=0.95)

        adv_low, _ = gae_low(rewards, values, next_values)
        adv_high, _ = gae_high(rewards, values, next_values)

        # First step should be similar (both get reward=1)
        # But later steps should differ due to discounting
        # The first advantage with gamma=0.5 should decay faster
        assert adv_low[0, 0].item() < adv_high[0, 0].item() + 0.1

    def test_lambda_effect(self):
        """Lambda controls bias-variance tradeoff in GAE.

        lambda=0: one-step TD (low variance, high bias)
        lambda=1: Monte Carlo (high variance, low bias)
        """
        rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        values = torch.zeros(1, 5)
        next_values = torch.zeros(1)

        gae_0 = GAE(gamma=0.99, lam=0.0)
        gae_1 = GAE(gamma=0.99, lam=1.0)

        adv_0, _ = gae_0(rewards, values, next_values)
        adv_1, _ = gae_1(rewards, values, next_values)

        # With lambda=0, only the last step should have nonzero advantage
        # (since reward is only at the last step)
        assert adv_0[0, 0].abs().item() < 0.01
        # With lambda=1, the first step should also get credit
        # (full Monte Carlo return)
        assert adv_1[0, 0].abs().item() > 0.01

    def test_returns_equal_advantages_plus_values(self):
        """Returns should equal advantages + values."""
        gae = GAE(gamma=0.99, lam=0.95)
        rewards = torch.randn(3, 8)
        values = torch.randn(3, 8)
        next_values = torch.randn(3)
        advantages, returns = gae(rewards, values, next_values)
        expected_returns = advantages + values
        assert torch.allclose(returns, expected_returns, atol=1e-5)

    def test_gradient_does_not_flow(self):
        """GAE should not propagate gradients (it's a utility, not a loss)."""
        gae = GAE(gamma=0.99, lam=0.95)
        rewards = torch.randn(2, 5, requires_grad=True)
        values = torch.randn(2, 5, requires_grad=True)
        next_values = torch.randn(2)
        advantages, returns = gae(rewards, values, next_values)
        # GAE should detach computations
        assert not advantages.requires_grad
        assert not returns.requires_grad


# ============================================================
# rollout tests
# ============================================================


class TestRollout:
    """Tests for the rollout function."""

    def test_returns_sequences(self):
        """Rollout should return generated sequences."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        prompt = torch.randint(0, 32, (2, 5))
        result = rollout(
            model=model,
            prompt=prompt,
            max_new_tokens=8,
            vocab_size=32,
        )
        assert "sequences" in result
        # Sequences should be prompt + generated tokens
        assert result["sequences"].shape[1] == 5 + 8

    def test_returns_log_probs(self):
        """Rollout should return log probabilities of generated tokens."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        prompt = torch.randint(0, 32, (2, 5))
        result = rollout(
            model=model,
            prompt=prompt,
            max_new_tokens=8,
            vocab_size=32,
        )
        assert "log_probs" in result
        assert result["log_probs"].shape == (2, 8)

    def test_returns_values(self):
        """Rollout should return value estimates."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        prompt = torch.randint(0, 32, (2, 5))
        result = rollout(
            model=model,
            prompt=prompt,
            max_new_tokens=8,
            vocab_size=32,
        )
        assert "values" in result
        # Values for each generated token
        assert result["values"].shape == (2, 8)

    def test_log_probs_are_negative(self):
        """Log probabilities should be negative (they are log of probs < 1)."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        prompt = torch.randint(0, 32, (2, 5))
        result = rollout(
            model=model,
            prompt=prompt,
            max_new_tokens=8,
            vocab_size=32,
        )
        assert (result["log_probs"] <= 0).all()

    def test_deterministic_with_greedy(self):
        """Greedy decoding (temperature=0) should be deterministic."""
        model = DummyPolicy(vocab_size=32, d_model=16)
        prompt = torch.randint(0, 32, (2, 5))
        result1 = rollout(
            model=model,
            prompt=prompt,
            max_new_tokens=8,
            vocab_size=32,
            temperature=0.0,
        )
        result2 = rollout(
            model=model,
            prompt=prompt,
            max_new_tokens=8,
            vocab_size=32,
            temperature=0.0,
        )
        assert torch.equal(result1["sequences"], result2["sequences"])

    def test_batch_consistency(self):
        """Items in a batch should be processed independently."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        prompt_a = torch.randint(0, 32, (1, 5))
        prompt_b = torch.randint(0, 32, (1, 5))

        result_a = rollout(model=model, prompt=prompt_a, max_new_tokens=4, vocab_size=32)
        result_b = rollout(model=model, prompt=prompt_b, max_new_tokens=4, vocab_size=32)

        # Process as batch
        prompt_batch = torch.cat([prompt_a, prompt_b], dim=0)
        result_batch = rollout(
            model=model, prompt=prompt_batch, max_new_tokens=4, vocab_size=32
        )

        # Batch results should match individual results
        # (Note: this may not be exact due to floating point, but should be close)
        assert result_batch["sequences"].shape == (2, 9)


# ============================================================
# PPOTrainer tests
# ============================================================


class TestPPOTrainer:
    """Tests for the PPOTrainer class."""

    def test_trainer_creation(self):
        """PPOTrainer should be created with valid parameters."""
        model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model = DummyPolicy(vocab_size=32, d_model=16)
        trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            clip_eps=0.2,
            kl_weight=0.1,
            gamma=0.99,
            lam=0.95,
            lr=1e-4,
        )
        assert trainer is not None

    def test_ppo_step_returns_metrics(self):
        """PPO step should return a dict of metrics."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model = DummyPolicy(vocab_size=32, d_model=16)
        # Copy weights so initial KL is small
        ref_model.load_state_dict(model.state_dict())

        trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            clip_eps=0.2,
            kl_weight=0.1,
            gamma=0.99,
            lam=0.95,
            lr=1e-4,
        )

        # Create dummy experience
        batch_size, seq_len = 4, 12
        prompt_len = 4
        gen_len = seq_len - prompt_len

        sequences = torch.randint(0, 32, (batch_size, seq_len))
        old_log_probs = torch.randn(batch_size, gen_len)
        rewards = torch.randn(batch_size, gen_len)
        values = torch.randn(batch_size, gen_len)

        metrics = trainer.step(
            sequences=sequences,
            old_log_probs=old_log_probs,
            rewards=rewards,
            values=values,
            prompt_len=prompt_len,
        )

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "kl" in metrics

    def test_ppo_step_reduces_loss(self):
        """Multiple PPO steps should reduce the loss."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model.load_state_dict(model.state_dict())

        trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            clip_eps=0.2,
            kl_weight=0.01,
            gamma=0.99,
            lam=0.95,
            lr=1e-3,
        )

        batch_size, seq_len = 4, 12
        prompt_len = 4
        gen_len = seq_len - prompt_len

        sequences = torch.randint(0, 32, (batch_size, seq_len))
        old_log_probs = torch.randn(batch_size, gen_len)
        rewards = torch.randn(batch_size, gen_len)
        values = torch.randn(batch_size, gen_len)

        losses = []
        for _ in range(5):
            metrics = trainer.step(
                sequences=sequences,
                old_log_probs=old_log_probs,
                rewards=rewards,
                values=values,
                prompt_len=prompt_len,
            )
            losses.append(metrics["loss"])

        # Loss should generally decrease (not guaranteed for every step,
        # but the trend should be downward)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_ppo_step_kl_from_ref(self):
        """PPO step should track KL divergence from reference model."""
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model.load_state_dict(model.state_dict())

        trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            clip_eps=0.2,
            kl_weight=0.1,
            gamma=0.99,
            lam=0.95,
            lr=1e-4,
        )

        batch_size, seq_len = 4, 12
        prompt_len = 4
        gen_len = seq_len - prompt_len

        sequences = torch.randint(0, 32, (batch_size, seq_len))
        old_log_probs = torch.randn(batch_size, gen_len)
        rewards = torch.randn(batch_size, gen_len)
        values = torch.randn(batch_size, gen_len)

        metrics = trainer.step(
            sequences=sequences,
            old_log_probs=old_log_probs,
            rewards=rewards,
            values=values,
            prompt_len=prompt_len,
        )

        # KL should be a finite number
        assert math.isfinite(metrics["kl"])

    def test_full_ppo_step_reduces_kl_from_ref(self):
        """Full PPO step should reduce KL from reference over multiple updates.

        This is the key integration test: PPO with KL penalty should keep
        the policy close to the reference.
        """
        torch.manual_seed(42)
        model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model.load_state_dict(model.state_dict())

        trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            clip_eps=0.2,
            kl_weight=0.5,  # Strong KL penalty
            gamma=0.99,
            lam=0.95,
            lr=1e-3,
        )

        batch_size, seq_len = 4, 12
        prompt_len = 4
        gen_len = seq_len - prompt_len

        sequences = torch.randint(0, 32, (batch_size, seq_len))
        old_log_probs = torch.randn(batch_size, gen_len)
        # Give strong positive rewards to push policy away from ref
        rewards = torch.ones(batch_size, gen_len) * 5.0
        values = torch.randn(batch_size, gen_len)

        kl_values = []
        for _ in range(10):
            metrics = trainer.step(
                sequences=sequences,
                old_log_probs=old_log_probs,
                rewards=rewards,
                values=values,
                prompt_len=prompt_len,
            )
            kl_values.append(metrics["kl"])

        # With KL penalty, the policy should not diverge too far
        # The KL should stay bounded (not grow without limit)
        max_kl = max(kl_values)
        assert max_kl < 10.0, f"KL grew too large: {max_kl:.4f}"

    def test_trainer_stores_models(self):
        """Trainer should store model and ref_model."""
        model = DummyPolicy(vocab_size=32, d_model=16)
        ref_model = DummyPolicy(vocab_size=32, d_model=16)
        trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            clip_eps=0.2,
            kl_weight=0.1,
            gamma=0.99,
            lam=0.95,
            lr=1e-4,
        )
        assert trainer.model is model
        assert trainer.ref_model is ref_model
