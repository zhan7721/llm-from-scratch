"""Tests for GRPO (Group Relative Policy Optimization) implementation."""

import torch
import torch.nn as nn
import pytest
import math
from grpo import (
    GRPOLoss,
    compute_group_advantages,
    GRPOTrainer,
    grpo_step,
    approx_kl_divergence,
    masked_mean,
)


class DummyGRPOPolicy(nn.Module):
    """Minimal policy model for GRPO testing.

    Unlike PPO's DummyPolicy, this has no value head -- GRPO doesn't
    need a value network since advantages come from reward normalization.
    """

    def __init__(self, vocab_size=32, d_model=16):
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

    def forward(self, input_ids):
        """Forward pass returning logits only (no value head).

        Args:
            input_ids: (batch, seq) token IDs.

        Returns:
            logits: (batch, seq, vocab_size)
        """
        h = self.embedding(input_ids)
        h = self.encoder(h)
        logits = self.policy_head(h)
        return logits


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
        assert (kl >= -1e-6).all()

    def test_kl_with_mask(self):
        """KL should respect the action mask."""
        log_probs = torch.randn(2, 8)
        log_probs_ref = log_probs + 1.0
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
        kl = approx_kl_divergence(log_probs, log_probs_ref, action_mask=mask)
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
        expected = torch.tensor([1.5])
        assert torch.allclose(result, expected)

    def test_full_mask(self):
        """Full mask should give regular mean."""
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1, 1, 1]])
        result = masked_mean(tensor, mask=mask, dim=-1)
        expected = torch.tensor([2.0])
        assert torch.allclose(result, expected)


# ============================================================
# compute_group_advantages tests
# ============================================================


class TestComputeGroupAdvantages:
    """Tests for the compute_group_advantages function."""

    def test_advantages_sum_to_zero(self):
        """Group advantages should sum to zero within each group.

        This is the defining property of GRPO: by normalizing rewards
        within a group, the advantages are zero-centered.
        """
        rewards = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 10.0],
            [-1.0, 0.0, 1.0, 2.0],
        ])
        advantages = compute_group_advantages(rewards)

        row_sums = advantages.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros(3), atol=1e-6), (
            f"Group advantages should sum to zero, got {row_sums}"
        )

    def test_advantages_zero_mean(self):
        """Advantages should have zero mean within each group."""
        torch.manual_seed(42)
        rewards = torch.randn(5, 8)
        advantages = compute_group_advantages(rewards)

        means = advantages.mean(dim=1)
        assert torch.allclose(means, torch.zeros(5), atol=1e-6)

    def test_advantages_unit_std(self):
        """Advantages should have unit std within each group."""
        torch.manual_seed(42)
        rewards = torch.randn(5, 8)
        advantages = compute_group_advantages(rewards)

        stds = advantages.std(dim=1)
        assert torch.allclose(stds, torch.ones(5), atol=1e-5), (
            f"Advantages should have unit std, got {stds}"
        )

    def test_higher_reward_higher_advantage(self):
        """Within a group, higher reward should give higher advantage."""
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        advantages = compute_group_advantages(rewards)

        for i in range(advantages.shape[1] - 1):
            assert advantages[0, i] < advantages[0, i + 1]

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        rewards = torch.randn(3, 4)
        advantages = compute_group_advantages(rewards)
        assert advantages.shape == rewards.shape

    def test_eps_prevents_division_by_zero(self):
        """eps parameter should prevent division by zero when all rewards are equal."""
        rewards = torch.tensor([[5.0, 5.0, 5.0]])
        advantages = compute_group_advantages(rewards)
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-3)

    def test_large_group(self):
        """Should work with larger groups."""
        torch.manual_seed(42)
        rewards = torch.randn(10, 16)
        advantages = compute_group_advantages(rewards)

        row_sums = advantages.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros(10), atol=1e-5)


# ============================================================
# GRPOLoss tests
# ============================================================


class TestGRPOLoss:
    """Tests for the GRPOLoss class."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar tensor."""
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=0.1)
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        log_probs_ref = torch.randn(4, 10)
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_kl_returned(self):
        """Should return KL divergence as second element."""
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=0.1)
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        log_probs_ref = torch.randn(4, 10)
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert isinstance(kl, torch.Tensor)

    def test_gradient_flows_through_policy(self):
        """Gradients should flow through the loss to the policy log probs."""
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=0.1)
        log_probs = torch.randn(4, 10, requires_grad=True)
        old_log_probs = torch.randn(4, 10).detach()
        advantages = torch.randn(4, 10)
        log_probs_ref = torch.randn(4, 10).detach()
        loss, _ = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        loss.backward()
        assert log_probs.grad is not None
        assert log_probs.grad.abs().sum() > 0

    def test_clip_bounds_advantage(self):
        """Clipped loss should bound the surrogate objective."""
        torch.manual_seed(42)
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=0.0)

        log_probs = torch.randn(2, 8) + 2.0
        old_log_probs = torch.randn(2, 8)
        advantages = torch.ones(2, 8) * 5.0
        log_probs_ref = old_log_probs.clone()

        loss_clipped, _ = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)

        max_surrogate = (1 + 0.2) * advantages.mean()
        assert (-loss_clipped).item() <= max_surrogate.item() + 1e-4

    def test_loss_positive_with_kl(self):
        """Loss should include KL penalty."""
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=1.0)
        log_probs = torch.randn(4, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.zeros(4, 10)
        log_probs_ref = log_probs + 1.0
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert loss.item() > 0

    def test_kl_weight_scales_penalty(self):
        """Higher kl_weight should increase the loss when KL > 0."""
        log_probs = torch.randn(4, 10)
        old_log_probs = log_probs.clone()
        advantages = torch.zeros(4, 10)
        log_probs_ref = log_probs + 0.5

        loss_low, _ = GRPOLoss(clip_eps=0.2, kl_weight=0.1)(
            log_probs, old_log_probs, advantages, log_probs_ref
        )
        loss_high, _ = GRPOLoss(clip_eps=0.2, kl_weight=1.0)(
            log_probs, old_log_probs, advantages, log_probs_ref
        )
        assert loss_high.item() > loss_low.item()

    def test_with_action_mask(self):
        """Loss should respect action mask."""
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=0.1)
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
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=1.0)
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        log_probs_ref = log_probs.clone()
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert kl.abs().max().item() < 1e-5

    def test_zero_advantage_zero_policy_loss(self):
        """With zero advantages, the policy loss component should be zero."""
        loss_fn = GRPOLoss(clip_eps=0.2, kl_weight=0.0)
        log_probs = torch.randn(4, 10)
        old_log_probs = torch.randn(4, 10)
        advantages = torch.zeros(4, 10)
        log_probs_ref = log_probs.clone()
        loss, kl = loss_fn(log_probs, old_log_probs, advantages, log_probs_ref)
        assert loss.abs().item() < 1e-6


# ============================================================
# grpo_step tests
# ============================================================


class TestGRPOStep:
    """Tests for the grpo_step function."""

    def _prepare_data(self):
        """Helper to set up model, ref_model, and data for grpo_step."""
        torch.manual_seed(42)
        model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        ref_model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sequences = torch.randint(0, 32, (4, 12))

        with torch.no_grad():
            logits = model(sequences)
            log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
            tokens = sequences[:, 1:].unsqueeze(-1)
            old_log_probs = log_probs_all[:, :-1, :].gather(-1, tokens).squeeze(-1)
            ref_logits = ref_model(sequences)
            ref_log_probs_all = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_probs_all[:, :-1, :].gather(-1, tokens).squeeze(-1)

        return model, ref_model, sequences, old_log_probs, ref_log_probs, optimizer

    def test_returns_metrics(self):
        """grpo_step should return a dict with loss and kl."""
        model, ref_model, sequences, old_lp, ref_lp, optimizer = self._prepare_data()
        advantages = torch.randn(4)

        metrics = grpo_step(
            model=model,
            ref_model=ref_model,
            sequences=sequences,
            old_log_probs=old_lp,
            ref_log_probs=ref_lp,
            advantages=advantages,
            optimizer=optimizer,
        )

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "kl" in metrics

    def test_loss_is_finite(self):
        """Loss should be a finite number."""
        model, ref_model, sequences, old_lp, ref_lp, optimizer = self._prepare_data()
        advantages = torch.randn(4)

        metrics = grpo_step(
            model=model,
            ref_model=ref_model,
            sequences=sequences,
            old_log_probs=old_lp,
            ref_log_probs=ref_lp,
            advantages=advantages,
            optimizer=optimizer,
        )

        assert math.isfinite(metrics["loss"])
        assert math.isfinite(metrics["kl"])

    def test_gradient_flows(self):
        """Gradient should flow through the policy after grpo_step."""
        model, ref_model, sequences, old_lp, ref_lp, optimizer = self._prepare_data()
        advantages = torch.randn(4)

        # Before the step, no gradients
        has_grad_before = any(p.grad is not None for p in model.parameters())
        assert not has_grad_before

        grpo_step(
            model=model,
            ref_model=ref_model,
            sequences=sequences,
            old_log_probs=old_lp,
            ref_log_probs=ref_lp,
            advantages=advantages,
            optimizer=optimizer,
        )

        # After the step, gradients should exist
        has_grad_after = any(p.grad is not None for p in model.parameters())
        assert has_grad_after

    def test_1d_advantages_expanded(self):
        """1D advantages should be expanded to match log_probs shape."""
        model, ref_model, sequences, old_lp, ref_lp, optimizer = self._prepare_data()
        advantages = torch.randn(4)

        metrics = grpo_step(
            model=model,
            ref_model=ref_model,
            sequences=sequences,
            old_log_probs=old_lp,
            ref_log_probs=ref_lp,
            advantages=advantages,
            optimizer=optimizer,
        )

        assert math.isfinite(metrics["loss"])


# ============================================================
# GRPOTrainer tests
# ============================================================


class TestGRPOTrainer:
    """Tests for the GRPOTrainer class."""

    def _make_reward_fn(self):
        """Create a simple reward function for testing."""
        def reward_fn(responses):
            if isinstance(responses[0], torch.Tensor):
                rewards = torch.tensor([float(r.shape[0]) for r in responses])
            else:
                rewards = torch.tensor([float(len(str(r))) for r in responses])
            return rewards
        return reward_fn

    def test_trainer_creation(self):
        """GRPOTrainer should be created with valid parameters."""
        model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        ref_model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        reward_fn = self._make_reward_fn()

        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            reward_fn=reward_fn,
            clip_eps=0.2,
            kl_weight=0.1,
            lr=1e-4,
        )
        assert trainer is not None

    def test_trainer_stores_models(self):
        """Trainer should store model and ref_model."""
        model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        ref_model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        reward_fn = self._make_reward_fn()

        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            reward_fn=reward_fn,
        )
        assert trainer.model is model
        assert trainer.ref_model is ref_model

    def test_trainer_freezes_ref_model(self):
        """Reference model should be frozen (no gradients)."""
        model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        ref_model = DummyGRPOPolicy(vocab_size=32, d_model=16)
        reward_fn = self._make_reward_fn()

        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            reward_fn=reward_fn,
        )

        for param in trainer.ref_model.parameters():
            assert not param.requires_grad
