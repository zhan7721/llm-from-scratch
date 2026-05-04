"""Tests for Process Reward Model (PRM) implementation."""

import torch
import torch.nn as nn
import pytest
import math
from process_reward_model import (
    ProcessRewardModel,
    StepwiseRewardDataset,
    StepwiseRewardLoss,
    best_of_n_prm,
)


class DummyTransformer(nn.Module):
    """Minimal transformer for PRM testing.

    Returns hidden states of shape (batch, seq_len, d_model) when given
    input_ids of shape (batch, seq_len). Has a `d_model` attribute
    that ProcessRewardModel uses to infer the hidden dimension.
    """

    def __init__(self, vocab_size=32, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, input_ids):
        """Return hidden states (batch, seq_len, d_model)."""
        h = self.embedding(input_ids)
        h = self.encoder(h)
        return h


# ============================================================
# ProcessRewardModel tests
# ============================================================


class TestProcessRewardModel:
    """Tests for the ProcessRewardModel class."""

    def _make_model_and_data(self, batch_size=2, seq_len=12, num_steps=3):
        """Helper to create a PRM and dummy data."""
        torch.manual_seed(42)
        transformer = DummyTransformer(vocab_size=32, d_model=16)
        prm = ProcessRewardModel(transformer)

        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        # Step boundaries: end position of each step
        # e.g., step 1 ends at token 3, step 2 at token 7, step 3 at token 11
        step_boundaries = torch.tensor([
            [3, 7, 11],
            [4, 8, 11],
        ][:batch_size])

        return prm, input_ids, step_boundaries

    def test_output_shape(self):
        """Per-step scores should have shape (batch, num_steps)."""
        prm, input_ids, step_boundaries = self._make_model_and_data()
        scores = prm(input_ids, step_boundaries)
        assert scores.shape == (2, 3), f"Expected (2, 3), got {scores.shape}"

    def test_output_is_finite(self):
        """All output scores should be finite."""
        prm, input_ids, step_boundaries = self._make_model_and_data()
        scores = prm(input_ids, step_boundaries)
        assert torch.isfinite(scores).all(), f"Non-finite scores: {scores}"

    def test_gradient_flows(self):
        """Gradients should flow through the PRM to transformer parameters."""
        prm, input_ids, step_boundaries = self._make_model_and_data()
        scores = prm(input_ids, step_boundaries)
        loss = scores.sum()
        loss.backward()

        # Check that at least some transformer parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in prm.parameters()
        )
        assert has_grad, "No gradients flowed through the PRM"

    def test_single_step(self):
        """PRM should work with a single step per sequence."""
        prm, input_ids, _ = self._make_model_and_data()
        step_boundaries = torch.tensor([[11], [11]])
        scores = prm(input_ids, step_boundaries)
        assert scores.shape == (2, 1)

    def test_many_steps(self):
        """PRM should work with many steps."""
        torch.manual_seed(42)
        transformer = DummyTransformer(vocab_size=32, d_model=16)
        prm = ProcessRewardModel(transformer)
        input_ids = torch.randint(0, 32, (1, 20))
        step_boundaries = torch.tensor([[2, 5, 8, 11, 14, 17, 19]])
        scores = prm(input_ids, step_boundaries)
        assert scores.shape == (1, 7)

    def test_eval_mode(self):
        """PRM should work in inference mode."""
        prm, input_ids, step_boundaries = self._make_model_and_data()
        prm.training = False
        with torch.no_grad():
            scores = prm(input_ids, step_boundaries)
        assert scores.shape == (2, 3)

    def test_batch_size_one(self):
        """PRM should work with batch_size=1."""
        prm, _, step_boundaries = self._make_model_and_data()
        input_ids = torch.randint(0, 32, (1, 12))
        step_boundaries = torch.tensor([[3, 7, 11]])
        scores = prm(input_ids, step_boundaries)
        assert scores.shape == (1, 3)

    def test_different_d_model(self):
        """PRM should work with different hidden dimensions."""
        for d_model in [8, 16, 32]:
            transformer = DummyTransformer(vocab_size=32, d_model=d_model)
            prm = ProcessRewardModel(transformer)
            input_ids = torch.randint(0, 32, (2, 10))
            step_boundaries = torch.tensor([[4, 9], [5, 9]])
            scores = prm(input_ids, step_boundaries)
            assert scores.shape == (2, 2)


# ============================================================
# StepwiseRewardDataset tests
# ============================================================


class TestStepwiseRewardDataset:
    """Tests for the StepwiseRewardDataset class."""

    def _make_dataset(self):
        """Helper to create a small dataset."""
        data = [
            {
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "step_boundaries": [3, 7],
                "step_labels": [1, 0],
            },
            {
                "input_ids": [10, 11, 12, 13, 14, 15],
                "step_boundaries": [2, 5],
                "step_labels": [1, 1],
            },
            {
                "input_ids": [20, 21, 22, 23, 24, 25, 26],
                "step_boundaries": [3, 6],
                "step_labels": [0, 0],
            },
        ]
        return StepwiseRewardDataset(data), data

    def test_len(self):
        """Dataset length should match the number of examples."""
        dataset, data = self._make_dataset()
        assert len(dataset) == len(data)

    def test_getitem_keys(self):
        """Each item should have the correct keys."""
        dataset, _ = self._make_dataset()
        item = dataset[0]
        assert "input_ids" in item
        assert "step_boundaries" in item
        assert "step_labels" in item

    def test_getitem_shapes(self):
        """Tensors should have correct shapes."""
        dataset, _ = self._make_dataset()
        item = dataset[0]
        assert item["input_ids"].dim() == 1
        assert item["step_boundaries"].dim() == 1
        assert item["step_labels"].dim() == 1
        assert item["step_boundaries"].shape[0] == item["step_labels"].shape[0]

    def test_step_labels_are_binary(self):
        """Step labels should be 0 or 1."""
        dataset, _ = self._make_dataset()
        for i in range(len(dataset)):
            labels = dataset[i]["step_labels"]
            assert ((labels == 0) | (labels == 1)).all(), (
                f"Labels should be 0 or 1, got {labels}"
            )

    def test_input_ids_are_long(self):
        """input_ids should be long integers."""
        dataset, _ = self._make_dataset()
        item = dataset[0]
        assert item["input_ids"].dtype == torch.long

    def test_step_boundaries_are_long(self):
        """step_boundaries should be long integers."""
        dataset, _ = self._make_dataset()
        item = dataset[0]
        assert item["step_boundaries"].dtype == torch.long

    def test_different_num_steps(self):
        """Dataset should handle examples with different numbers of steps."""
        data = [
            {
                "input_ids": [1, 2, 3],
                "step_boundaries": [2],
                "step_labels": [1],
            },
            {
                "input_ids": [4, 5, 6, 7, 8],
                "step_boundaries": [2, 4],
                "step_labels": [1, 0],
            },
        ]
        dataset = StepwiseRewardDataset(data)
        assert dataset[0]["step_labels"].shape[0] == 1
        assert dataset[1]["step_labels"].shape[0] == 2


# ============================================================
# StepwiseRewardLoss tests
# ============================================================


class TestStepwiseRewardLoss:
    """Tests for the StepwiseRewardLoss class."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar tensor."""
        loss_fn = StepwiseRewardLoss()
        scores = torch.randn(4, 3)
        labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]).float()
        loss = loss_fn(scores, labels)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_loss_with_perfect_logits(self):
        """Loss should be near 0 when logits are very confident and correct."""
        loss_fn = StepwiseRewardLoss()
        # Very high logits for label=1, very low for label=0
        scores = torch.tensor([[10.0, -10.0, 10.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])
        loss = loss_fn(scores, labels)
        assert loss.item() < 0.01, f"Expected near-zero loss, got {loss.item()}"

    def test_loss_with_wrong_logits(self):
        """Loss should be high when logits are confidently wrong."""
        loss_fn = StepwiseRewardLoss()
        # Very high logits for label=0, very low for label=1
        scores = torch.tensor([[10.0, -10.0]])
        labels = torch.tensor([[0.0, 1.0]])
        loss = loss_fn(scores, labels)
        assert loss.item() > 5.0, f"Expected high loss, got {loss.item()}"

    def test_loss_non_negative(self):
        """Loss should always be non-negative."""
        loss_fn = StepwiseRewardLoss()
        scores = torch.randn(8, 5)
        labels = torch.randint(0, 2, (8, 5)).float()
        loss = loss_fn(scores, labels)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        """Gradients should flow through the loss to the scores."""
        loss_fn = StepwiseRewardLoss()
        scores = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]).float()
        loss = loss_fn(scores, labels)
        loss.backward()
        assert scores.grad is not None
        assert scores.grad.abs().sum() > 0

    def test_loss_with_mask(self):
        """Loss should respect step_mask when provided."""
        loss_fn = StepwiseRewardLoss()
        scores = torch.tensor([[5.0, -5.0, 5.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])
        # Mask out the last step
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss_masked = loss_fn(scores, labels, step_mask=mask)

        # Compare with loss computed on only the first two steps
        scores_subset = torch.tensor([[5.0, -5.0]])
        labels_subset = torch.tensor([[1.0, 0.0]])
        loss_subset = loss_fn(scores_subset, labels_subset)

        assert torch.allclose(loss_masked, loss_subset, atol=1e-6), (
            f"Masked loss {loss_masked.item()} should equal subset loss {loss_subset.item()}"
        )

    def test_uniform_predictions(self):
        """Loss should be ~ln(2) when logits are 0 (50% confidence)."""
        loss_fn = StepwiseRewardLoss()
        scores = torch.zeros(4, 3)
        labels = torch.randint(0, 2, (4, 3)).float()
        loss = loss_fn(scores, labels)
        expected = math.log(2)
        assert abs(loss.item() - expected) < 0.01, (
            f"Expected ~{expected:.4f}, got {loss.item():.4f}"
        )


# ============================================================
# best_of_n_prm tests
# ============================================================


class TestBestOfNPrm:
    """Tests for the best_of_n_prm function."""

    def test_selects_highest_min_score(self):
        """Should select candidate with highest min step score (default)."""
        # Candidate 0: steps [0.9, 0.1, 0.9] -> min = 0.1
        # Candidate 1: steps [0.5, 0.6, 0.7] -> min = 0.5
        # Candidate 2: steps [0.3, 0.8, 0.4] -> min = 0.3
        step_scores = torch.tensor([
            [0.9, 0.1, 0.9],
            [0.5, 0.6, 0.7],
            [0.3, 0.8, 0.4],
        ])
        best = best_of_n_prm(step_scores, aggregation='min')
        assert best == 1, f"Expected candidate 1, got {best}"

    def test_selects_highest_sum_score(self):
        """Should select candidate with highest sum of step scores."""
        # Candidate 0: sum = 1.9
        # Candidate 1: sum = 1.5
        # Candidate 2: sum = 1.8
        step_scores = torch.tensor([
            [0.9, 0.1, 0.9],
            [0.5, 0.5, 0.5],
            [0.3, 0.8, 0.7],
        ])
        best = best_of_n_prm(step_scores, aggregation='sum')
        assert best == 0, f"Expected candidate 0, got {best}"

    def test_selects_highest_mean_score(self):
        """Should select candidate with highest mean step score."""
        step_scores = torch.tensor([
            [0.2, 0.2, 0.2],  # mean = 0.2
            [0.8, 0.8, 0.8],  # mean = 0.8
            [0.5, 0.5, 0.5],  # mean = 0.5
        ])
        best = best_of_n_prm(step_scores, aggregation='mean')
        assert best == 1, f"Expected candidate 1, got {best}"

    def test_selects_highest_product_score(self):
        """Should select candidate with highest product of step scores."""
        # Use probabilities (0 to 1) for product to make sense
        step_scores = torch.tensor([
            [0.9, 0.9, 0.9],  # product = 0.729
            [0.5, 0.5, 0.5],  # product = 0.125
            [0.8, 0.8, 0.8],  # product = 0.512
        ])
        best = best_of_n_prm(step_scores, aggregation='product')
        assert best == 0, f"Expected candidate 0, got {best}"

    def test_single_candidate(self):
        """With one candidate, should return 0."""
        step_scores = torch.tensor([[0.5, 0.6, 0.7]])
        best = best_of_n_prm(step_scores)
        assert best == 0

    def test_returns_int(self):
        """Return value should be a plain Python int."""
        step_scores = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
        best = best_of_n_prm(step_scores)
        assert isinstance(best, int)

    def test_with_mask(self):
        """Should ignore padded steps when mask is provided."""
        # Candidate 0: 2 valid steps [0.9, 0.1], rest padded
        # Candidate 1: 3 valid steps [0.5, 0.6, 0.7]
        step_scores = torch.tensor([
            [0.9, 0.1, 0.0],  # min of valid = 0.1
            [0.5, 0.6, 0.7],  # min = 0.5
        ])
        mask = torch.tensor([
            [1, 1, 0],
            [1, 1, 1],
        ]).float()
        best = best_of_n_prm(step_scores, aggregation='min', step_mask=mask)
        assert best == 1, f"Expected candidate 1, got {best}"

    def test_with_mask_sum(self):
        """Sum aggregation should respect the mask."""
        step_scores = torch.tensor([
            [1.0, 1.0, 0.0],  # sum of valid = 2.0
            [0.5, 0.5, 0.5],  # sum = 1.5
        ])
        mask = torch.tensor([
            [1, 1, 0],
            [1, 1, 1],
        ]).float()
        best = best_of_n_prm(step_scores, aggregation='sum', step_mask=mask)
        assert best == 0, f"Expected candidate 0, got {best}"

    def test_two_candidates(self):
        """Should correctly compare two candidates."""
        step_scores = torch.tensor([
            [0.3, 0.3],
            [0.8, 0.8],
        ])
        best = best_of_n_prm(step_scores, aggregation='min')
        assert best == 1

    def test_tie_returns_first(self):
        """When tied, should return the first candidate."""
        step_scores = torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5],
        ])
        best = best_of_n_prm(step_scores, aggregation='sum')
        assert best == 0
