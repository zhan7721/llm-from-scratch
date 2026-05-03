import torch
import pytest
from data_pipeline import PretrainDataset, PackedDataset, dynamic_pad_collate, create_pretrain_dataloader


def test_pretrain_dataset_length():
    dataset = PretrainDataset(list(range(100)), seq_len=10)
    assert len(dataset) == 10


def test_pretrain_dataset_shapes():
    dataset = PretrainDataset(list(range(100)), seq_len=10)
    item = dataset[0]
    assert item["input_ids"].shape == (10,)
    assert item["labels"].shape == (10,)


def test_dataset_input_label_shift():
    """Labels should be input shifted by 1 position."""
    tokens = list(range(21))
    dataset = PretrainDataset(tokens, seq_len=10)
    item = dataset[0]
    assert item["input_ids"][1].item() == item["labels"][0].item()


def test_dynamic_pad_collate():
    batch = [
        {"input_ids": torch.arange(5), "labels": torch.arange(5)},
        {"input_ids": torch.arange(8), "labels": torch.arange(8)},
    ]
    result = dynamic_pad_collate(batch, pad_id=0)
    assert result["input_ids"].shape == (2, 8)
    assert result["attention_mask"][0, 5:].sum() == 0  # padded positions
    assert result["attention_mask"][0, :5].sum() == 5  # real positions


def test_dynamic_pad_collate_ignore_index():
    batch = [
        {"input_ids": torch.arange(5), "labels": torch.arange(5)},
        {"input_ids": torch.arange(8), "labels": torch.arange(8)},
    ]
    result = dynamic_pad_collate(batch, pad_id=0)
    assert (result["labels"][0, 5:] == -100).all()  # padded labels ignored


def test_packed_dataset_no_waste():
    """Packed dataset should use all tokens with no padding waste."""
    tokens = list(range(100))
    dataset = PackedDataset([tokens], seq_len=10, pad_id=0)
    assert len(dataset) > 0
    # All items should be full length
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["input_ids"].shape == (10,)


def test_packed_dataset_shapes():
    tokens = list(range(100))
    dataset = PackedDataset([tokens], seq_len=10, pad_id=0)
    item = dataset[0]
    assert item["input_ids"].shape == (10,)
    assert item["labels"].shape == (10,)


def test_create_pretrain_dataloader():
    tokens = list(range(200))
    loader = create_pretrain_dataloader(tokens, seq_len=10, batch_size=4)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4  # batch size
    assert batch["input_ids"].shape[1] == 10  # seq_len


def test_create_pretrain_dataloader_packed():
    tokens = list(range(200))
    loader = create_pretrain_dataloader(tokens, seq_len=10, batch_size=4, use_packing=True)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape[1] == 10
