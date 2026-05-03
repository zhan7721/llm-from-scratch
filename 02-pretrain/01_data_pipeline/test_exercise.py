"""Tests for the Data Pipeline exercise.

These tests are skipped by default. As you implement each method in
exercise.py, remove the `@pytest.mark.skip` decorator from the
corresponding test to verify your work.
"""

import torch
import pytest
from exercise import PretrainDatasetExercise, PackedDatasetExercise, dynamic_pad_collate_exercise


@pytest.mark.skip(reason="Enable after implementing PretrainDataset.__getitem__")
def test_exercise_pretrain_dataset_length():
    dataset = PretrainDatasetExercise(list(range(100)), seq_len=10)
    assert len(dataset) == 10


@pytest.mark.skip(reason="Enable after implementing PretrainDataset.__getitem__")
def test_exercise_pretrain_dataset_shapes():
    dataset = PretrainDatasetExercise(list(range(100)), seq_len=10)
    item = dataset[0]
    assert item["input_ids"].shape == (10,)
    assert item["labels"].shape == (10,)


@pytest.mark.skip(reason="Enable after implementing PretrainDataset.__getitem__")
def test_exercise_dataset_input_label_shift():
    """Labels should be input shifted by 1 position."""
    tokens = list(range(21))
    dataset = PretrainDatasetExercise(tokens, seq_len=10)
    item = dataset[0]
    assert item["input_ids"][1].item() == item["labels"][0].item()


@pytest.mark.skip(reason="Enable after implementing dynamic_pad_collate")
def test_exercise_dynamic_pad_collate():
    batch = [
        {"input_ids": torch.arange(5), "labels": torch.arange(5)},
        {"input_ids": torch.arange(8), "labels": torch.arange(8)},
    ]
    result = dynamic_pad_collate_exercise(batch, pad_id=0)
    assert result["input_ids"].shape == (2, 8)
    assert result["attention_mask"][0, 5:].sum() == 0
    assert result["attention_mask"][0, :5].sum() == 5


@pytest.mark.skip(reason="Enable after implementing dynamic_pad_collate")
def test_exercise_dynamic_pad_collate_ignore_index():
    batch = [
        {"input_ids": torch.arange(5), "labels": torch.arange(5)},
        {"input_ids": torch.arange(8), "labels": torch.arange(8)},
    ]
    result = dynamic_pad_collate_exercise(batch, pad_id=0)
    assert (result["labels"][0, 5:] == -100).all()


@pytest.mark.skip(reason="Enable after implementing PackedDataset.__getitem__")
def test_exercise_packed_dataset_shapes():
    tokens = list(range(100))
    dataset = PackedDatasetExercise([tokens], seq_len=10, pad_id=0)
    item = dataset[0]
    assert item["input_ids"].shape == (10,)
    assert item["labels"].shape == (10,)


@pytest.mark.skip(reason="Enable after implementing PackedDataset.__getitem__")
def test_exercise_packed_dataset_no_waste():
    tokens = list(range(100))
    dataset = PackedDatasetExercise([tokens], seq_len=10, pad_id=0)
    assert len(dataset) > 0
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["input_ids"].shape == (10,)
