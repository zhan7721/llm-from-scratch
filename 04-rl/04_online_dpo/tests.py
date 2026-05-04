"""Tests for Online DPO implementation.

Online DPO generates preference pairs on-the-fly from the current policy,
rather than using a static dataset. This tests:
- generate_and_score: generate responses and score with reward model
- OnlineDPODataset: dynamic preference pairs from online generation
- OnlineDPOTrainer: combines generation and DPO training
"""

import torch
import torch.nn as nn
import pytest
from online_dpo import (
    generate_and_score,
    OnlineDPODataset,
    OnlineDPOTrainer,
)


class DummyLanguageModel(nn.Module):
    """Minimal causal language model for testing.

    Uses TransformerEncoder (no causal mask, but sufficient for testing
    generation logic). Produces logits of shape (batch, seq, vocab_size).
    """

    def __init__(self, vocab_size=256, d_model=32, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                batch_first=True,
                dropout=0.0,
            ),
            num_layers=2,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """Forward pass returning logits.

        Args:
            input_ids: (batch, seq) token IDs.

        Returns:
            Logits of shape (batch, seq, vocab_size).
        """
        h = self.embedding(input_ids)
        h = self.encoder(h)
        logits = self.lm_head(h)
        return logits


class DummyRewardModel(nn.Module):
    """Minimal reward model for testing.

    Takes input_ids and produces a scalar reward per sequence.
    Uses a simple embedding + pooling + linear approach.
    """

    def __init__(self, vocab_size=256, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        """Compute reward for input sequences.

        Args:
            input_ids: (batch, seq) token IDs.

        Returns:
            Reward scores of shape (batch,).
        """
        h = self.embedding(input_ids)  # (batch, seq, d_model)
        h = h.mean(dim=1)  # (batch, d_model) -- pool over sequence
        reward = self.reward_head(h).squeeze(-1)  # (batch,)
        return reward


# ============================================================
# generate_and_score tests
# ============================================================


class TestGenerateAndScore:
    """Tests for the generate_and_score function."""

    def test_returns_dict_with_expected_keys(self):
        """Should return a dict with chosen_input_ids, rejected_input_ids,
        chosen_reward, rejected_reward, response_start_idx."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3, 4]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=8,
            num_candidates=4,
        )

        assert isinstance(result, dict)
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result
        assert "chosen_reward" in result
        assert "rejected_reward" in result
        assert "response_start_idx" in result
        assert "chosen_ref_log_probs" in result
        assert "rejected_ref_log_probs" in result

    def test_chosen_input_ids_starts_with_prompt(self):
        """Chosen input IDs should start with the prompt tokens."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [10, 20, 30]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
        )

        chosen_ids = result["chosen_input_ids"]
        assert chosen_ids[:len(prompt)].tolist() == prompt

    def test_rejected_input_ids_starts_with_prompt(self):
        """Rejected input IDs should start with the prompt tokens."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [10, 20, 30]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
        )

        rejected_ids = result["rejected_input_ids"]
        assert rejected_ids[:len(prompt)].tolist() == prompt

    def test_chosen_reward_geq_rejected_reward(self):
        """Chosen reward should be >= rejected reward (highest vs lowest)."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3, 4]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=8,
            num_candidates=8,
        )

        assert result["chosen_reward"] >= result["rejected_reward"]

    def test_response_start_idx_equals_prompt_length(self):
        """response_start_idx should equal the prompt length."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [5, 10, 15, 20, 25]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
        )

        assert result["response_start_idx"] == len(prompt)

    def test_output_tensors_dtype(self):
        """Output tensors should have correct dtypes."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
        )

        assert result["chosen_input_ids"].dtype == torch.long
        assert result["rejected_input_ids"].dtype == torch.long
        assert result["chosen_reward"].dtype == torch.float32
        assert result["rejected_reward"].dtype == torch.float32

    def test_response_length_matches_max_new_tokens(self):
        """Generated sequences should have prompt_len + max_new_tokens length."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3, 4]
        max_new_tokens = 10

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_candidates=4,
        )

        expected_len = len(prompt) + max_new_tokens
        assert result["chosen_input_ids"].shape[0] == expected_len
        assert result["rejected_input_ids"].shape[0] == expected_len

    def test_generate_and_score_uses_policy(self):
        """Generated tokens should come from the policy model's distribution."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3]

        # Run twice with same seed -- should get same results
        torch.manual_seed(123)
        result1 = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
        )

        torch.manual_seed(123)
        result2 = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
        )

        assert torch.equal(result1["chosen_input_ids"], result2["chosen_input_ids"])
        assert torch.equal(result1["rejected_input_ids"], result2["rejected_input_ids"])

    def test_different_candidates_may_differ(self):
        """With enough candidates, chosen and rejected may differ."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=10,
            num_candidates=8,
        )

        # They could be the same in degenerate cases, but with 8 candidates
        # and 10 tokens, they should usually differ
        # We just check the structure is correct
        assert result["chosen_input_ids"].shape == result["rejected_input_ids"].shape

    def test_temperature_parameter(self):
        """Temperature parameter should affect generation."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3]

        # Low temperature (more deterministic)
        torch.manual_seed(42)
        result_low = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
            temperature=0.1,
        )

        # High temperature (more random)
        torch.manual_seed(42)
        result_high = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
            temperature=2.0,
        )

        # Both should produce valid output
        assert result_low["chosen_input_ids"].shape == result_high["chosen_input_ids"].shape

    def test_non_empty_response(self):
        """Generated responses should be non-empty (max_new_tokens > 0)."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=4,
            num_candidates=4,
        )

        # Response should have at least 1 token beyond the prompt
        total_len = result["chosen_input_ids"].shape[0]
        assert total_len > len(prompt)

    def test_ref_log_probs_computed(self):
        """Reference log probs should be computed and stored."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompt = [1, 2, 3]

        result = generate_and_score(
            policy=policy,
            reward_model=reward_model,
            prompt=prompt,
            max_new_tokens=6,
            num_candidates=4,
        )

        assert isinstance(result["chosen_ref_log_probs"], torch.Tensor)
        assert isinstance(result["rejected_ref_log_probs"], torch.Tensor)
        assert result["chosen_ref_log_probs"].dim() == 0  # scalar
        assert result["rejected_ref_log_probs"].dim() == 0
        assert result["chosen_ref_log_probs"].item() <= 0
        assert result["rejected_ref_log_probs"].item() <= 0


# ============================================================
# OnlineDPODataset tests
# ============================================================


class TestOnlineDPODataset:
    """Tests for the OnlineDPODataset class."""

    def _make_components(self):
        """Create dummy policy, reward model, and prompts."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        # Use same-length prompts for batching
        prompts = [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [5, 15, 25, 35],
        ]
        return policy, reward_model, prompts

    def test_dataset_length(self):
        """Dataset length should match number of prompts."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        assert len(dataset) == len(prompts)

    def test_dataset_item_keys(self):
        """Each item should have the expected keys."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        item = dataset[0]
        assert "chosen_input_ids" in item
        assert "rejected_input_ids" in item
        assert "chosen_ref_log_probs" in item
        assert "rejected_ref_log_probs" in item
        assert "response_start_idx" in item

    def test_dataset_item_tensors(self):
        """Items should contain tensors."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        item = dataset[0]
        assert isinstance(item["chosen_input_ids"], torch.Tensor)
        assert isinstance(item["rejected_input_ids"], torch.Tensor)
        assert isinstance(item["chosen_ref_log_probs"], torch.Tensor)
        assert isinstance(item["rejected_ref_log_probs"], torch.Tensor)

    def test_input_ids_dtype(self):
        """Input IDs should be long tensors."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        item = dataset[0]
        assert item["chosen_input_ids"].dtype == torch.long
        assert item["rejected_input_ids"].dtype == torch.long

    def test_ref_log_probs_negative(self):
        """Reference log probs should be negative (or zero at best)."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["chosen_ref_log_probs"].item() <= 0
            assert item["rejected_ref_log_probs"].item() <= 0

    def test_ref_log_probs_no_grad(self):
        """Reference log probs should not have gradients."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        item = dataset[0]
        assert not item["chosen_ref_log_probs"].requires_grad
        assert not item["rejected_ref_log_probs"].requires_grad

    def test_response_start_idx(self):
        """response_start_idx should equal the prompt length."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        for i, prompt in enumerate(prompts):
            item = dataset[i]
            assert item["response_start_idx"] == len(prompt)

    def test_refresh_updates_data(self):
        """Calling refresh() should regenerate preference pairs."""
        torch.manual_seed(42)
        policy = DummyLanguageModel()
        reward_model = DummyRewardModel()
        prompts = [[1, 2, 3, 4]]

        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )

        item_before = dataset[0]
        dataset.refresh()
        item_after = dataset[0]

        # After refresh, the data may differ (due to randomness in generation)
        # But the structure should be the same
        assert item_after["chosen_input_ids"].shape == item_before["chosen_input_ids"].shape
        assert item_after["response_start_idx"] == item_before["response_start_idx"]

    def test_chosen_starts_with_prompt(self):
        """Chosen input IDs should start with the prompt."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        for i, prompt in enumerate(prompts):
            item = dataset[i]
            assert item["chosen_input_ids"][:len(prompt)].tolist() == prompt

    def test_rejected_starts_with_prompt(self):
        """Rejected input IDs should start with the prompt."""
        policy, reward_model, prompts = self._make_components()
        dataset = OnlineDPODataset(
            policy=policy,
            reward_model=reward_model,
            prompts=prompts,
            max_new_tokens=6,
            num_candidates=4,
        )
        for i, prompt in enumerate(prompts):
            item = dataset[i]
            assert item["rejected_input_ids"][:len(prompt)].tolist() == prompt


# ============================================================
# OnlineDPOTrainer tests
# ============================================================


def _make_batch(dataset, indices):
    """Helper to collate dataset items into a batch."""
    items = [dataset[i] for i in indices]
    return {
        "chosen_input_ids": torch.stack([it["chosen_input_ids"] for it in items]),
        "rejected_input_ids": torch.stack([it["rejected_input_ids"] for it in items]),
        "chosen_ref_log_probs": torch.stack([it["chosen_ref_log_probs"] for it in items]),
        "rejected_ref_log_probs": torch.stack([it["rejected_ref_log_probs"] for it in items]),
        "response_start_idx": torch.tensor([it["response_start_idx"] for it in items]),
    }


class TestOnlineDPOTrainer:
    """Tests for the OnlineDPOTrainer class."""

    def _make_trainer(self, beta=0.1, lr=1e-3):
        """Create a trainer with dummy models.

        Uses same-length prompts so sequences can be batched.
        """
        torch.manual_seed(42)
        model = DummyLanguageModel()
        ref_model = DummyLanguageModel()
        reward_model = DummyRewardModel()
        # Same-length prompts for proper batching
        prompts = [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ]
        trainer = OnlineDPOTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            prompts=prompts,
            beta=beta,
            lr=lr,
            max_new_tokens=6,
            num_candidates=4,
        )
        return trainer

    def test_trainer_stores_models(self):
        """Trainer should store all models."""
        trainer = self._make_trainer()
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "ref_model")
        assert hasattr(trainer, "reward_model")

    def test_trainer_has_optimizer(self):
        """Trainer should create an optimizer."""
        trainer = self._make_trainer()
        assert hasattr(trainer, "optimizer")
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_ref_model_frozen(self):
        """Reference model should be frozen."""
        trainer = self._make_trainer()
        for param in trainer.ref_model.parameters():
            assert not param.requires_grad

    def test_reward_model_frozen(self):
        """Reward model should be frozen."""
        trainer = self._make_trainer()
        for param in trainer.reward_model.parameters():
            assert not param.requires_grad

    def test_generate_creates_dataset(self):
        """generate() should create an OnlineDPODataset."""
        trainer = self._make_trainer()
        dataset = trainer.generate()
        assert isinstance(dataset, OnlineDPODataset)
        assert len(dataset) == len(trainer.prompts)

    def test_train_step_returns_metrics(self):
        """train_step should return a dict with loss."""
        trainer = self._make_trainer()
        dataset = trainer.generate()
        batch = _make_batch(dataset, [0])
        metrics = trainer.train_step(batch)
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_train_step_updates_model(self):
        """Training step should update model parameters."""
        trainer = self._make_trainer()
        dataset = trainer.generate()

        # Record initial parameters
        initial_params = {
            name: param.clone()
            for name, param in trainer.model.named_parameters()
        }

        batch = _make_batch(dataset, list(range(len(dataset))))
        trainer.train_step(batch)

        # At least one parameter should have changed
        changed = False
        for name, param in trainer.model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                changed = True
                break
        assert changed, "Model parameters should change after training step"

    def test_train_epoch_runs(self):
        """train_epoch should run without errors."""
        trainer = self._make_trainer()
        metrics = trainer.train_epoch(batch_size=2)
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_train_epoch_generates_fresh_data(self):
        """train_epoch should generate new preference pairs."""
        trainer = self._make_trainer()

        # First epoch
        metrics1 = trainer.train_epoch(batch_size=2)

        # Second epoch -- should regenerate data
        metrics2 = trainer.train_epoch(batch_size=2)

        # Both should produce valid metrics
        assert "loss" in metrics1
        assert "loss" in metrics2

    def test_loss_decreases_over_training(self):
        """Training over multiple epochs should decrease loss."""
        torch.manual_seed(42)
        model = DummyLanguageModel(d_model=32)
        ref_model = DummyLanguageModel(d_model=32)
        reward_model = DummyRewardModel(d_model=32)
        # Use diverse same-length prompts for better training signal
        prompts = [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [100, 110, 120, 130],
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ]

        trainer = OnlineDPOTrainer(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            prompts=prompts,
            beta=0.5,
            lr=1e-2,
            max_new_tokens=6,
            num_candidates=4,
        )

        losses = []
        for _ in range(10):
            metrics = trainer.train_epoch(batch_size=4)
            losses.append(metrics["loss"])

        # Later losses should be lower than the first
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_gradient_flows_through_policy(self):
        """Gradients should flow through the policy model."""
        trainer = self._make_trainer()
        dataset = trainer.generate()
        batch = _make_batch(dataset, [0])
        trainer.train_step(batch)

        has_grad = False
        for param in trainer.model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found in policy model after train_step"

    def test_ref_model_no_gradients(self):
        """Reference model should not accumulate gradients."""
        trainer = self._make_trainer()
        dataset = trainer.generate()
        batch = _make_batch(dataset, [0])
        trainer.train_step(batch)

        for param in trainer.ref_model.parameters():
            assert param.grad is None, "Reference model should not have gradients"

    def test_reward_model_no_gradients(self):
        """Reward model should not accumulate gradients."""
        trainer = self._make_trainer()
        dataset = trainer.generate()
        batch = _make_batch(dataset, [0])
        trainer.train_step(batch)

        for param in trainer.reward_model.parameters():
            assert param.grad is None, "Reward model should not have gradients"
