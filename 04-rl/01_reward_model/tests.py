"""Tests for Reward Model implementation."""

import torch
import torch.nn as nn
import pytest
from reward_model import (
    RewardModel,
    BradleyTerryLoss,
    RewardDataset,
    train_reward_model,
)


class DummyTransformer(nn.Module):
    """Minimal transformer-like model for testing.

    Produces hidden states of shape (batch, seq, d_model).
    This simulates the output of a real transformer.
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
        """Forward pass returning hidden states.

        Args:
            input_ids: (batch, seq) token IDs.

        Returns:
            Hidden states of shape (batch, seq, d_model).
        """
        h = self.embedding(input_ids)
        h = self.encoder(h)
        return h


# ============================================================
# RewardModel tests
# ============================================================


class TestRewardModel:
    """Tests for the RewardModel class."""

    def test_output_shape_scalar(self):
        """RewardModel should output one scalar per input in the batch."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        input_ids = torch.randint(0, 256, (3, 16))
        rewards = model(input_ids)
        assert rewards.shape == (3,), f"Expected shape (3,), got {rewards.shape}"

    def test_output_shape_single(self):
        """Single input should produce a scalar."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        input_ids = torch.randint(0, 256, (1, 10))
        rewards = model(input_ids)
        assert rewards.shape == (1,), f"Expected shape (1,), got {rewards.shape}"

    def test_output_is_float(self):
        """Output should be a float tensor."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        input_ids = torch.randint(0, 256, (2, 12))
        rewards = model(input_ids)
        assert rewards.dtype == torch.float32, f"Expected float32, got {rewards.dtype}"

    def test_different_inputs_give_different_rewards(self):
        """Different random inputs should generally produce different rewards."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        input_a = torch.randint(0, 256, (1, 10))
        input_b = torch.randint(0, 256, (1, 10))
        reward_a = model(input_a)
        reward_b = model(input_b)
        # Not guaranteed but very likely with random init
        assert not torch.allclose(reward_a, reward_b, atol=1e-6)

    def test_gradient_flows(self):
        """Gradients should flow through the reward model."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        input_ids = torch.randint(0, 256, (2, 10))
        rewards = model(input_ids)
        loss = rewards.sum()
        loss.backward()
        # Check that embedding gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_wraps_transformer(self):
        """RewardModel should store the transformer as an attribute."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        assert model.transformer is transformer

    def test_has_scalar_head(self):
        """RewardModel should have a scalar head (Linear -> 1 output)."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        assert hasattr(model, "scalar_head")
        assert isinstance(model.scalar_head, nn.Linear)
        assert model.scalar_head.out_features == 1

    def test_batch_consistency(self):
        """Processing a batch should give the same results as processing items individually."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        model.eval()
        input_a = torch.randint(0, 256, (1, 10))
        input_b = torch.randint(0, 256, (1, 10))
        batch = torch.cat([input_a, input_b], dim=0)
        batch_rewards = model(batch)
        reward_a = model(input_a)
        reward_b = model(input_b)
        assert torch.allclose(batch_rewards[0:1], reward_a, atol=1e-5)
        assert torch.allclose(batch_rewards[1:2], reward_b, atol=1e-5)


# ============================================================
# BradleyTerryLoss tests
# ============================================================


class TestBradleyTerryLoss:
    """Tests for the BradleyTerryLoss class."""

    def test_loss_is_scalar(self):
        """Loss should be a scalar tensor."""
        loss_fn = BradleyTerryLoss()
        chosen = torch.tensor([1.0, 2.0])
        rejected = torch.tensor([0.0, 1.0])
        loss = loss_fn(chosen, rejected)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_loss_positive(self):
        """Loss should always be non-negative."""
        loss_fn = BradleyTerryLoss()
        chosen = torch.tensor([1.0, 2.0, 3.0])
        rejected = torch.tensor([0.0, 0.5, 1.0])
        loss = loss_fn(chosen, rejected)
        assert loss.item() >= 0

    def test_loss_decreases_with_better_ranking(self):
        """Loss should be smaller when chosen > rejected by a larger margin."""
        loss_fn = BradleyTerryLoss()
        # Good ranking: chosen much higher than rejected
        chosen_good = torch.tensor([5.0, 5.0])
        rejected_good = torch.tensor([0.0, 0.0])
        loss_good = loss_fn(chosen_good, rejected_good)

        # Bad ranking: chosen and rejected are close
        chosen_bad = torch.tensor([1.1, 1.1])
        rejected_bad = torch.tensor([1.0, 1.0])
        loss_bad = loss_fn(chosen_bad, rejected_bad)

        assert loss_good.item() < loss_bad.item()

    def test_loss_zero_when_perfect(self):
        """Loss approaches zero when chosen >> rejected."""
        loss_fn = BradleyTerryLoss()
        chosen = torch.tensor([100.0])
        rejected = torch.tensor([-100.0])
        loss = loss_fn(chosen, rejected)
        assert loss.item() < 0.01

    def test_loss_high_when_wrong_order(self):
        """Loss should be high when rejected > chosen."""
        loss_fn = BradleyTerryLoss()
        chosen = torch.tensor([-10.0])
        rejected = torch.tensor([10.0])
        loss = loss_fn(chosen, rejected)
        assert loss.item() > 5.0  # -log(sigmoid(-20)) is large

    def test_gradient_flows(self):
        """Gradients should flow through the loss."""
        loss_fn = BradleyTerryLoss()
        chosen = torch.tensor([2.0], requires_grad=True)
        rejected = torch.tensor([1.0], requires_grad=True)
        loss = loss_fn(chosen, rejected)
        loss.backward()
        assert chosen.grad is not None
        assert rejected.grad is not None

    def test_equal_rewards_gives_log2(self):
        """When chosen == rejected, loss = -log(0.5) = log(2)."""
        loss_fn = BradleyTerryLoss()
        chosen = torch.tensor([3.0, 3.0])
        rejected = torch.tensor([3.0, 3.0])
        loss = loss_fn(chosen, rejected)
        expected = torch.log(torch.tensor(2.0))
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_batch_loss(self):
        """Loss should average over the batch."""
        loss_fn = BradleyTerryLoss()
        chosen = torch.tensor([5.0, 0.0])
        rejected = torch.tensor([0.0, 5.0])
        loss = loss_fn(chosen, rejected)
        # Individual losses: -log(sigmoid(5)) and -log(sigmoid(-5))
        l1 = -torch.log(torch.sigmoid(torch.tensor(5.0)))
        l2 = -torch.log(torch.sigmoid(torch.tensor(-5.0)))
        expected = (l1 + l2) / 2
        assert torch.allclose(loss, expected, atol=1e-5)


# ============================================================
# RewardDataset tests
# ============================================================


class TestRewardDataset:
    """Tests for the RewardDataset class."""

    def test_dataset_length(self):
        """Dataset length should match number of pairs."""
        pairs = [
            {"chosen": [1, 2, 3], "rejected": [4, 5, 6]},
            {"chosen": [7, 8, 9], "rejected": [10, 11, 12]},
        ]
        dataset = RewardDataset(pairs)
        assert len(dataset) == 2

    def test_dataset_item_keys(self):
        """Each item should have chosen_input_ids and rejected_input_ids."""
        pairs = [{"chosen": [1, 2, 3], "rejected": [4, 5, 6]}]
        dataset = RewardDataset(pairs)
        item = dataset[0]
        assert "chosen_input_ids" in item
        assert "rejected_input_ids" in item

    def test_dataset_item_tensors(self):
        """Items should be tensors."""
        pairs = [{"chosen": [1, 2, 3], "rejected": [4, 5, 6]}]
        dataset = RewardDataset(pairs)
        item = dataset[0]
        assert isinstance(item["chosen_input_ids"], torch.Tensor)
        assert isinstance(item["rejected_input_ids"], torch.Tensor)

    def test_dataset_item_dtypes(self):
        """Tensors should be long (for embedding lookup)."""
        pairs = [{"chosen": [1, 2, 3], "rejected": [4, 5, 6]}]
        dataset = RewardDataset(pairs)
        item = dataset[0]
        assert item["chosen_input_ids"].dtype == torch.long
        assert item["rejected_input_ids"].dtype == torch.long

    def test_dataset_preserves_values(self):
        """Dataset should preserve the token IDs from the input."""
        pairs = [{"chosen": [10, 20, 30], "rejected": [40, 50, 60]}]
        dataset = RewardDataset(pairs)
        item = dataset[0]
        assert item["chosen_input_ids"].tolist() == [10, 20, 30]
        assert item["rejected_input_ids"].tolist() == [40, 50, 60]

    def test_dataset_with_padding(self):
        """Dataset should support padding to max_length."""
        pairs = [
            {"chosen": [1, 2], "rejected": [3, 4, 5, 6]},
        ]
        dataset = RewardDataset(pairs, max_length=8, pad_token_id=0)
        item = dataset[0]
        assert item["chosen_input_ids"].shape[0] == 8
        assert item["rejected_input_ids"].shape[0] == 8
        # chosen should be padded
        assert item["chosen_input_ids"].tolist() == [1, 2, 0, 0, 0, 0, 0, 0]

    def test_dataset_attention_mask(self):
        """Dataset should produce attention masks when padding is used."""
        pairs = [{"chosen": [1, 2], "rejected": [3, 4, 5, 6]}]
        dataset = RewardDataset(pairs, max_length=8, pad_token_id=0)
        item = dataset[0]
        assert "chosen_attention_mask" in item
        assert "rejected_attention_mask" in item
        # chosen: 2 real tokens + 6 padding
        assert item["chosen_attention_mask"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
        # rejected: 4 real tokens + 4 padding
        assert item["rejected_attention_mask"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


# ============================================================
# train_reward_model tests
# ============================================================


class TestTrainRewardModel:
    """Tests for the train_reward_model function."""

    def test_loss_decreases(self):
        """Training should decrease the loss over iterations."""
        torch.manual_seed(42)
        transformer = DummyTransformer(d_model=32, vocab_size=256)
        model = RewardModel(transformer)

        # Create preference data where chosen always has higher token IDs
        pairs = []
        for _ in range(50):
            chosen = torch.randint(100, 200, (10,)).tolist()
            rejected = torch.randint(0, 100, (10,)).tolist()
            pairs.append({"chosen": chosen, "rejected": rejected})

        dataset = RewardDataset(pairs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        initial_loss = None
        final_loss = None

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = BradleyTerryLoss()

        model.train()
        for epoch in range(5):
            for batch in dataloader:
                loss = train_reward_model(model, batch, loss_fn, optimizer)
                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_preference_ordering(self):
        """After training, chosen examples should get higher rewards than rejected."""
        torch.manual_seed(42)
        transformer = DummyTransformer(d_model=32, vocab_size=256)
        model = RewardModel(transformer)

        # Create clear preference signal
        pairs = []
        for _ in range(100):
            chosen = torch.randint(150, 256, (10,)).tolist()
            rejected = torch.randint(0, 100, (10,)).tolist()
            pairs.append({"chosen": chosen, "rejected": rejected})

        dataset = RewardDataset(pairs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = BradleyTerryLoss()

        model.train()
        for epoch in range(10):
            for batch in dataloader:
                train_reward_model(model, batch, loss_fn, optimizer)

        # Evaluate: chosen should have higher rewards
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in dataloader:
                chosen_rewards = model(batch["chosen_input_ids"])
                rejected_rewards = model(batch["rejected_input_ids"])
                correct += (chosen_rewards > rejected_rewards).sum().item()
                total += chosen_rewards.size(0)

        accuracy = correct / total
        assert accuracy > 0.6, f"Accuracy too low: {accuracy:.2f}"

    def test_returns_loss(self):
        """train_reward_model should return the loss value."""
        transformer = DummyTransformer()
        model = RewardModel(transformer)
        batch = {
            "chosen_input_ids": torch.randint(0, 256, (2, 10)),
            "rejected_input_ids": torch.randint(0, 256, (2, 10)),
        }
        loss_fn = BradleyTerryLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = train_reward_model(model, batch, loss_fn, optimizer)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
