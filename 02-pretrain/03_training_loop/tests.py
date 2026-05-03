import torch
import torch.nn as nn
import pytest
from training_loop import (
    CosineLRScheduler, GradientClipper, TrainingMetrics,
    create_training_components, training_step,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 10)
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(self.embedding(x))


def test_cosine_lr_warmup():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineLRScheduler(optimizer, warmup_steps=10, total_steps=100)

    scheduler.step(0)
    assert optimizer.param_groups[0]["lr"] == 0.0  # step 0, scale = 0

    scheduler.step(5)
    assert 0 < optimizer.param_groups[0]["lr"] < 1e-3  # mid warmup


def test_cosine_lr_peak():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineLRScheduler(optimizer, warmup_steps=10, total_steps=100)

    scheduler.step(10)
    assert abs(optimizer.param_groups[0]["lr"] - 1e-3) < 1e-6  # at end of warmup


def test_cosine_lr_decay():
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineLRScheduler(optimizer, warmup_steps=10, total_steps=100)

    scheduler.step(10)
    peak_lr = optimizer.param_groups[0]["lr"]

    scheduler.step(100)
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr < peak_lr


def test_gradient_clipper():
    model = DummyModel()
    # Set large gradients
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 10.0

    clipper = GradientClipper(max_norm=1.0)
    norm = clipper.clip(model)
    assert norm > 1.0  # norm was above threshold

    # Check gradients are now clipped
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    assert total_norm <= 1.0 + 1e-6


def test_training_metrics():
    metrics = TrainingMetrics()
    metrics.update(2.5, 1.0, 1e-3)
    metrics.update(2.3, 0.8, 9e-4)

    assert abs(metrics.avg_loss - 2.4) < 0.01
    assert abs(metrics.avg_grad_norm - 0.9) < 0.01
    assert metrics.summary()["steps"] == 2


def test_create_training_components():
    model = DummyModel()
    components = create_training_components(model, warmup_steps=10, total_steps=100)
    assert "optimizer" in components
    assert "scheduler" in components
    assert "clipper" in components
    assert "metrics" in components


def test_training_step():
    model = DummyModel()
    components = create_training_components(model, warmup_steps=10, total_steps=100)

    batch = {
        "input_ids": torch.randint(0, 10, (2, 10)),
        "labels": torch.randint(0, 10, (2, 10)),
    }

    result = training_step(
        model, batch,
        components["optimizer"],
        components["scheduler"],
        components["clipper"],
        step=0,
    )

    assert "loss" in result
    assert "grad_norm" in result
    assert "lr" in result
    assert result["loss"] > 0
