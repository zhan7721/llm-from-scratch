"""Instruction tuning: format and fine-tune on instruction data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    """Dataset for instruction tuning.

    Each example has: instruction, optional input, output.
    Formats into a prompt template and tokenizes.
    """

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer=None,
        max_length: int = 512,
        template: str = "alpaca",
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

    def __len__(self) -> int:
        return len(self.examples)

    def _format_alpaca(self, example: Dict[str, str]) -> str:
        """Format in Alpaca style."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        return prompt + output

    def _format_prompt_only(self, example: Dict[str, str]) -> str:
        """Format only the prompt (for generation)."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")

        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        full_text = self._format_alpaca(example)
        prompt_text = self._format_prompt_only(example)

        if self.tokenizer is not None:
            full_ids = self.tokenizer.encode(full_text)[:self.max_length]
            prompt_ids = self.tokenizer.encode(prompt_text)[:self.max_length]
        else:
            # Simple character-level tokenization for testing
            full_ids = [ord(c) % 256 for c in full_text][:self.max_length]
            prompt_ids = [ord(c) % 256 for c in prompt_text][:self.max_length]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Mask instruction portion in labels (only train on response)
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
    template: str = "alpaca",
) -> str:
    """Format an instruction example into a prompt string.

    Args:
        instruction: The instruction text.
        input_text: Optional input context.
        output: The expected output.
        template: Template style ("alpaca" or "simple").

    Returns:
        Formatted string.
    """
    if template == "alpaca":
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt + output
    else:
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
        else:
            prompt = f"Instruction: {instruction}\nOutput: "
        return prompt + output


def compute_instruction_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute loss for instruction tuning.

    Only computes loss on the response portion (labels != -100).
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    outputs = model(input_ids)
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss
