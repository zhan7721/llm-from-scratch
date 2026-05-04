import torch
import pytest
from instruction_tuning import InstructionDataset, format_instruction, compute_instruction_loss
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.head(self.embed(x))


def test_dataset_length():
    examples = [
        {"instruction": "Summarize", "input": "Long text...", "output": "Summary."},
        {"instruction": "Translate", "input": "Hello", "output": "Hola"},
    ]
    dataset = InstructionDataset(examples)
    assert len(dataset) == 2


def test_dataset_item_shapes():
    examples = [{"instruction": "Test", "output": "Result"}]
    dataset = InstructionDataset(examples)
    item = dataset[0]
    assert "input_ids" in item
    assert "labels" in item
    assert "attention_mask" in item
    assert item["input_ids"].shape == item["labels"].shape


def test_dataset_masks_instruction():
    """Labels should be -100 for instruction portion."""
    examples = [{"instruction": "Do this", "output": "Done."}]
    dataset = InstructionDataset(examples, max_length=100)
    item = dataset[0]
    # Some labels should be -100 (masked)
    assert (item["labels"] == -100).any()
    # Some should be actual tokens (response)
    assert (item["labels"] != -100).any()


def test_format_instruction_alpaca():
    result = format_instruction("Summarize", "Long text", "Summary.", template="alpaca")
    assert "### Instruction:" in result
    assert "### Input:" in result
    assert "### Response:" in result
    assert "Summary." in result


def test_format_instruction_no_input():
    result = format_instruction("Write a poem", output="Roses are red...", template="alpaca")
    assert "### Instruction:" in result
    assert "### Input:" not in result
    assert "### Response:" in result


def test_format_instruction_simple():
    result = format_instruction("Test", output="Result", template="simple")
    assert "Instruction:" in result
    assert "Output:" in result


def test_compute_instruction_loss():
    model = DummyModel()
    batch = {
        "input_ids": torch.randint(0, 256, (2, 20)),
        "labels": torch.randint(0, 256, (2, 20)),
    }
    # Set some labels to -100
    batch["labels"][:, :10] = -100
    loss = compute_instruction_loss(model, batch)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_dataset_with_tokenizer():
    """Dataset should work with a simple tokenizer."""
    class SimpleTokenizer:
        def encode(self, text):
            return [ord(c) % 256 for c in text]

    examples = [{"instruction": "Test", "output": "OK"}]
    dataset = InstructionDataset(examples, tokenizer=SimpleTokenizer(), max_length=50)
    item = dataset[0]
    assert item["input_ids"].shape[0] <= 50
