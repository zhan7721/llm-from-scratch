"""Tests for DPO (Direct Preference Optimization) implementation."""

import torch
import torch.nn as nn
import pytest
from dpo import (
    compute_log_probs,
    DPOLoss,
    DPODataset,
    DPOTrainer,
)


class DummyLanguageModel(nn.Module):
    """Minimal language model for testing DPO.

    Produces logits of shape (batch, seq, vocab_size).
    This simulates the output of a real causal language model.
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


# ============================================================
# compute_log_probs tests
# ============================================================


class TestComputeLogProbs:
    """Tests for the compute_log_probs function."""

    def test_output_shape(self):
        """compute_log_probs should return shape (batch,)."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (3, 16))
        log_probs = compute_log_probs(model, input_ids, response_start_idx=8)
        assert log_probs.shape == (3,), f"Expected shape (3,), got {log_probs.shape}"

    def test_output_is_float(self):
        """Output should be a float tensor."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (2, 12))
        log_probs = compute_log_probs(model, input_ids, response_start_idx=6)
        assert log_probs.dtype == torch.float32, f"Expected float32, got {log_probs.dtype}"

    def test_output_is_negative(self):
        """Log probabilities should be negative (or zero at best)."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (4, 10))
        log_probs = compute_log_probs(model, input_ids, response_start_idx=5)
        assert (log_probs <= 0).all(), "Log probs should be <= 0"

    def test_sums_over_response_tokens(self):
        """Should sum log probs over response tokens, not prompt tokens."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (2, 12))
        log_probs_short = compute_log_probs(model, input_ids, response_start_idx=10)
        log_probs_long = compute_log_probs(model, input_ids, response_start_idx=6)
        assert log_probs_short.shape == (2,)
        assert log_probs_long.shape == (2,)

    def test_gradient_flows(self):
        """Gradients should flow through compute_log_probs."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (2, 10))
        log_probs = compute_log_probs(model, input_ids, response_start_idx=5)
        loss = log_probs.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic(self):
        """Same input should give same output (no randomness)."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (2, 10))
        with torch.no_grad():
            lp1 = compute_log_probs(model, input_ids, response_start_idx=5)
            lp2 = compute_log_probs(model, input_ids, response_start_idx=5)
        assert torch.allclose(lp1, lp2)

    def test_full_sequence_response(self):
        """When response_start_idx=1, all tokens except first are response."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (2, 10))
        log_probs = compute_log_probs(model, input_ids, response_start_idx=1)
        assert log_probs.shape == (2,)
        assert (log_probs < 0).all()

    def test_single_response_token(self):
        """When response has only one token, log prob is just that token's log prob."""
        model = DummyLanguageModel()
        input_ids = torch.randint(0, 256, (1, 6))
        with torch.no_grad():
            log_probs = compute_log_probs(model, input_ids, response_start_idx=5)
        assert log_probs.shape == (1,)
        assert log_probs.item() < 0


# ============================================================
# DPOLoss tests
# ============================================================


class TestDPOLoss:
    """Tests for the DPOLoss class."""

    def test_loss_is_scalar(self):
        """DPO loss should be a scalar tensor."""
        loss_fn = DPOLoss(beta=0.1)
        policy_chosen = torch.tensor([1.0, 2.0])
        policy_rejected = torch.tensor([0.0, 1.0])
        ref_chosen = torch.tensor([0.5, 1.5])
        ref_rejected = torch.tensor([0.0, 0.5])
        loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_loss_positive(self):
        """Loss should always be non-negative."""
        loss_fn = DPOLoss(beta=0.1)
        policy_chosen = torch.tensor([1.0, 2.0, 3.0])
        policy_rejected = torch.tensor([0.0, 0.5, 1.0])
        ref_chosen = torch.tensor([0.5, 1.0, 2.0])
        ref_rejected = torch.tensor([0.0, 0.3, 0.8])
        loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert loss.item() >= 0

    def test_preferred_gets_higher_reward(self):
        """Loss should be lower when policy assigns higher reward to chosen."""
        loss_fn = DPOLoss(beta=0.1)
        ref_chosen = torch.tensor([0.0, 0.0])
        ref_rejected = torch.tensor([0.0, 0.0])
        policy_good_chosen = torch.tensor([5.0, 5.0])
        policy_good_rejected = torch.tensor([0.0, 0.0])
        loss_good = loss_fn(policy_good_chosen, policy_good_rejected, ref_chosen, ref_rejected)

        policy_bad_chosen = torch.tensor([0.1, 0.1])
        policy_bad_rejected = torch.tensor([0.0, 0.0])
        loss_bad = loss_fn(policy_bad_chosen, policy_bad_rejected, ref_chosen, ref_rejected)

        assert loss_good.item() < loss_bad.item()

    def test_loss_zero_when_perfect(self):
        """Loss approaches zero when chosen >> rejected in implicit reward."""
        loss_fn = DPOLoss(beta=0.1)
        policy_chosen = torch.tensor([100.0])
        policy_rejected = torch.tensor([0.0])
        ref_chosen = torch.tensor([0.0])
        ref_rejected = torch.tensor([0.0])
        loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert loss.item() < 0.01

    def test_loss_high_when_wrong_order(self):
        """Loss should be high when implicit reward of rejected > chosen."""
        loss_fn = DPOLoss(beta=0.5)
        # With beta=0.5: implicit reward chosen = 0.5*(-10-0)=-5, rejected=0.5*(10-0)=5
        # loss = -logsigmoid(-5 - 5) = -logsigmoid(-10) ~ 10
        policy_chosen = torch.tensor([-10.0])
        policy_rejected = torch.tensor([10.0])
        ref_chosen = torch.tensor([0.0])
        ref_rejected = torch.tensor([0.0])
        loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert loss.item() > 5.0

    def test_gradient_flows(self):
        """Gradients should flow through the DPO loss."""
        loss_fn = DPOLoss(beta=0.1)
        policy_chosen = torch.tensor([2.0], requires_grad=True)
        policy_rejected = torch.tensor([1.0], requires_grad=True)
        ref_chosen = torch.tensor([0.5])
        ref_rejected = torch.tensor([0.3])
        loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        loss.backward()
        assert policy_chosen.grad is not None
        assert policy_rejected.grad is not None

    def test_beta_controls_sensitivity(self):
        """Higher beta should make the loss more sensitive to reward differences."""
        policy_chosen = torch.tensor([3.0])
        policy_rejected = torch.tensor([1.0])
        ref_chosen = torch.tensor([2.0])
        ref_rejected = torch.tensor([0.5])

        loss_low_beta = DPOLoss(beta=0.01)(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        loss_high_beta = DPOLoss(beta=1.0)(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

        assert loss_high_beta.item() < loss_low_beta.item()

    def test_equal_policy_ref_gives_log2(self):
        """When policy == reference, loss = -log(sigmoid(0)) = log(2)."""
        loss_fn = DPOLoss(beta=0.1)
        chosen = torch.tensor([3.0, 3.0])
        rejected = torch.tensor([3.0, 3.0])
        loss = loss_fn(chosen, rejected, chosen, rejected)
        expected = torch.log(torch.tensor(2.0))
        assert torch.allclose(loss, expected, atol=1e-5)


# ============================================================
# DPODataset tests
# ============================================================


class TestDPODataset:
    """Tests for the DPODataset class."""

    def _make_dummy_data(self, n=3, prompt_len=4, response_len=6, vocab_size=256):
        """Create dummy preference data for testing."""
        pairs = []
        for _ in range(n):
            prompt = torch.randint(0, vocab_size, (prompt_len,)).tolist()
            chosen = torch.randint(0, vocab_size, (response_len,)).tolist()
            rejected = torch.randint(0, vocab_size, (response_len,)).tolist()
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
        return pairs

    def test_dataset_length(self):
        """Dataset length should match number of pairs."""
        pairs = self._make_dummy_data(n=5)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        assert len(dataset) == 5

    def test_dataset_item_keys(self):
        """Each item should have the expected keys."""
        pairs = self._make_dummy_data(n=1)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item = dataset[0]
        assert "chosen_input_ids" in item
        assert "rejected_input_ids" in item
        assert "chosen_ref_log_probs" in item
        assert "rejected_ref_log_probs" in item
        assert "response_start_idx" in item

    def test_dataset_item_tensors(self):
        """Items should contain tensors."""
        pairs = self._make_dummy_data(n=1)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item = dataset[0]
        assert isinstance(item["chosen_input_ids"], torch.Tensor)
        assert isinstance(item["rejected_input_ids"], torch.Tensor)
        assert isinstance(item["chosen_ref_log_probs"], torch.Tensor)
        assert isinstance(item["rejected_ref_log_probs"], torch.Tensor)

    def test_input_ids_dtype(self):
        """Input IDs should be long tensors."""
        pairs = self._make_dummy_data(n=1)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item = dataset[0]
        assert item["chosen_input_ids"].dtype == torch.long
        assert item["rejected_input_ids"].dtype == torch.long

    def test_ref_log_probs_scalar(self):
        """Reference log probs should be scalars (summed over response)."""
        pairs = self._make_dummy_data(n=1)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item = dataset[0]
        assert item["chosen_ref_log_probs"].dim() == 0
        assert item["rejected_ref_log_probs"].dim() == 0

    def test_ref_log_probs_negative(self):
        """Reference log probs should be negative (or zero at best)."""
        pairs = self._make_dummy_data(n=3)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["chosen_ref_log_probs"].item() <= 0
            assert item["rejected_ref_log_probs"].item() <= 0

    def test_ref_log_probs_no_grad(self):
        """Reference log probs should not have gradients."""
        pairs = self._make_dummy_data(n=1)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item = dataset[0]
        assert not item["chosen_ref_log_probs"].requires_grad
        assert not item["rejected_ref_log_probs"].requires_grad

    def test_input_ids_concatenation(self):
        """Input IDs should be prompt + response concatenated."""
        prompt = [1, 2, 3]
        chosen = [10, 20]
        rejected = [30, 40]
        pairs = [{"prompt": prompt, "chosen": chosen, "rejected": rejected}]
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item = dataset[0]
        assert item["chosen_input_ids"][:3].tolist() == prompt
        assert item["chosen_input_ids"][3:].tolist() == chosen
        assert item["rejected_input_ids"][:3].tolist() == prompt
        assert item["rejected_input_ids"][3:].tolist() == rejected

    def test_response_start_idx(self):
        """response_start_idx should equal the prompt length."""
        prompt = [1, 2, 3, 4]
        chosen = [10, 20]
        rejected = [30, 40]
        pairs = [{"prompt": prompt, "chosen": chosen, "rejected": rejected}]
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item = dataset[0]
        assert item["response_start_idx"] == 4

    def test_ref_model_frozen(self):
        """Reference model should be frozen after dataset creation."""
        pairs = self._make_dummy_data(n=1)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        for param in ref_model.parameters():
            assert not param.requires_grad

    def test_consistent_ref_log_probs(self):
        """Same data should give same reference log probs."""
        pairs = self._make_dummy_data(n=1)
        ref_model = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model)
        item1 = dataset[0]
        item2 = dataset[0]
        assert torch.allclose(item1["chosen_ref_log_probs"], item2["chosen_ref_log_probs"])
        assert torch.allclose(item1["rejected_ref_log_probs"], item2["rejected_ref_log_probs"])


# ============================================================
# DPOTrainer tests
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


class TestDPOTrainer:
    """Tests for the DPOTrainer class."""

    def _make_dummy_dataset(self, n=4, prompt_len=4, response_len=6):
        """Create a dummy DPODataset for testing."""
        pairs = []
        for _ in range(n):
            prompt = torch.randint(0, 256, (prompt_len,)).tolist()
            chosen = torch.randint(0, 256, (response_len,)).tolist()
            rejected = torch.randint(0, 256, (response_len,)).tolist()
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
        ref_model = DummyLanguageModel()
        return DPODataset(pairs, ref_model)

    def test_trainer_stores_models(self):
        """Trainer should store policy and reference models."""
        model = DummyLanguageModel()
        ref_model = DummyLanguageModel()
        trainer = DPOTrainer(model, ref_model)
        assert trainer.model is model
        assert trainer.ref_model is ref_model

    def test_trainer_has_optimizer(self):
        """Trainer should create an optimizer."""
        model = DummyLanguageModel()
        ref_model = DummyLanguageModel()
        trainer = DPOTrainer(model, ref_model)
        assert hasattr(trainer, "optimizer")
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_ref_model_frozen(self):
        """Reference model should be frozen."""
        model = DummyLanguageModel()
        ref_model = DummyLanguageModel()
        trainer = DPOTrainer(model, ref_model)
        for param in trainer.ref_model.parameters():
            assert not param.requires_grad

    def test_step_returns_metrics(self):
        """Trainer.step should return a dict with metrics."""
        dataset = self._make_dummy_dataset(n=2)
        model = DummyLanguageModel()
        ref_model = DummyLanguageModel()
        trainer = DPOTrainer(model, ref_model, beta=0.1)
        batch = _make_batch(dataset, [0, 1])
        metrics = trainer.step(batch)
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_loss_decreases(self):
        """Training should decrease the loss over iterations."""
        torch.manual_seed(42)
        model = DummyLanguageModel(d_model=32)
        ref_model = DummyLanguageModel(d_model=32)

        pairs = []
        for _ in range(20):
            prompt = torch.randint(0, 256, (4,)).tolist()
            chosen = torch.randint(100, 256, (6,)).tolist()
            rejected = torch.randint(0, 100, (6,)).tolist()
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        ref_model_copy = DummyLanguageModel(d_model=32)
        ref_model_copy.load_state_dict(ref_model.state_dict())
        dataset = DPODataset(pairs, ref_model_copy)

        trainer = DPOTrainer(model, ref_model, beta=0.1, lr=1e-3)

        batch_size = 4
        initial_loss = None
        final_loss = None

        for epoch in range(5):
            for i in range(0, len(dataset), batch_size):
                indices = list(range(i, min(i + batch_size, len(dataset))))
                batch = _make_batch(dataset, indices)
                metrics = trainer.step(batch)
                if initial_loss is None:
                    initial_loss = metrics["loss"]
                final_loss = metrics["loss"]

        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_gradient_flows_through_policy(self):
        """Gradients should flow through the policy model."""
        model = DummyLanguageModel()
        ref_model = DummyLanguageModel()
        trainer = DPOTrainer(model, ref_model, beta=0.1)

        pairs = [{
            "prompt": [1, 2, 3, 4],
            "chosen": [10, 20, 30, 40, 50, 60],
            "rejected": [11, 21, 31, 41, 51, 61],
        }]
        ref_model_for_data = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model_for_data)
        batch = _make_batch(dataset, [0])
        trainer.step(batch)

        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found in policy model after step"

    def test_ref_model_no_gradients(self):
        """Reference model should not accumulate gradients."""
        model = DummyLanguageModel()
        ref_model = DummyLanguageModel()
        trainer = DPOTrainer(model, ref_model, beta=0.1)

        pairs = [{
            "prompt": [1, 2, 3, 4],
            "chosen": [10, 20, 30, 40, 50, 60],
            "rejected": [11, 21, 31, 41, 51, 61],
        }]
        ref_model_for_data = DummyLanguageModel()
        dataset = DPODataset(pairs, ref_model_for_data)
        batch = _make_batch(dataset, [0])
        trainer.step(batch)

        for param in ref_model.parameters():
            assert param.grad is None, "Reference model should not have gradients"
