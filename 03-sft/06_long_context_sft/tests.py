import torch
import torch.nn as nn
import pytest
from long_context_sft import (
    PositionInterpolation, NTKAwareScaling, LongContextDataset,
    compute_long_context_loss, prepare_long_context_batch, LongContextTrainer,
)


class DummyModel(nn.Module):
    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.head(self.embed(x))


def test_position_interpolation():
    pi = PositionInterpolation(original_max_seq_len=2048, target_max_seq_len=8192)
    positions = torch.arange(8192)
    rescaled = pi.rescale_positions(positions)
    assert rescaled.max() < 2048
    assert rescaled.min() >= 0


def test_ntk_scaling():
    ntk = NTKAwareScaling(original_max_seq_len=2048, target_max_seq_len=8192)
    inv_freq = ntk.get_scaled_inv_freq(d_model=64, device=torch.device("cpu"))
    assert inv_freq.shape == (32,)
    assert (inv_freq > 0).all()


def test_long_context_dataset():
    docs = [list(range(100)), list(range(200))]
    dataset = LongContextDataset(documents=docs, seq_len=50)
    assert len(dataset) > 0
    item = dataset[0]
    assert item["input_ids"].shape == (50,)
    assert item["labels"].shape == (50,)


def test_long_context_dataset_sliding_window():
    doc = list(range(200))
    dataset = LongContextDataset(documents=[doc], seq_len=50, stride=25)
    assert len(dataset) > 1  # Multiple windows


def test_long_context_dataset_short_doc():
    doc = list(range(20))  # shorter than seq_len
    dataset = LongContextDataset(documents=[doc], seq_len=50)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["input_ids"].shape == (50,)


def test_compute_long_context_loss():
    model = DummyModel()
    batch = {
        "input_ids": torch.randint(0, 100, (2, 100)),
        "labels": torch.randint(0, 100, (2, 100)),
    }
    loss = compute_long_context_loss(model, batch, chunk_size=50)
    assert loss.item() > 0


def test_compute_long_context_loss_short():
    model = DummyModel()
    batch = {
        "input_ids": torch.randint(0, 100, (2, 30)),
        "labels": torch.randint(0, 100, (2, 30)),
    }
    loss = compute_long_context_loss(model, batch, chunk_size=50)
    assert loss.item() > 0


def test_prepare_batch():
    batch = prepare_long_context_batch(list(range(100)), seq_len=50)
    assert batch["input_ids"].shape == (1, 50)
    assert batch["labels"].shape == (1, 50)
    assert batch["attention_mask"].shape == (1, 50)


def test_prepare_batch_short():
    batch = prepare_long_context_batch(list(range(20)), seq_len=50, pad_id=0)
    assert batch["input_ids"].shape == (1, 50)
    assert batch["attention_mask"][0, 20:].sum() == 0  # padded


def test_long_context_trainer():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = LongContextTrainer(model, optimizer, original_max_seq_len=64, target_max_seq_len=128)

    batch = {
        "input_ids": torch.randint(0, 100, (1, 50)),
        "labels": torch.randint(0, 100, (1, 50)),
    }
    result = trainer.train_step(batch, step=0)
    assert "loss" in result
    assert result["loss"] > 0
