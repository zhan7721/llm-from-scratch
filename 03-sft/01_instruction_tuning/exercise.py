"""Instruction Tuning -- Exercise.

Fill in the TODO methods to implement instruction tuning components.
Start with `format_instruction`, then `InstructionDataset.__getitem__`,
and finally `compute_instruction_loss`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from torch.utils.data import Dataset


class InstructionDatasetExercise(Dataset):
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
        """Get a single tokenized example with masked labels.

        Args:
            idx: Index into the examples list.

        Returns:
            Dict with "input_ids", "labels", "attention_mask".

        Hints:
            1. Get the example at index `idx`.
            2. Format the full text (instruction + response) using `_format_alpaca`.
            3. Format the prompt only (instruction without response) using `_format_prompt_only`.
            4. Tokenize both. If `self.tokenizer` is not None, use `self.tokenizer.encode(text)`.
               Otherwise, use a simple fallback: `[ord(c) % 256 for c in text]`.
            5. Truncate both to `self.max_length`.
            6. Create `input_ids` as a torch.long tensor of the full token IDs.
            7. Clone `input_ids` to create `labels`.
            8. Mask the instruction portion in labels: set `labels[:len(prompt_ids)] = -100`.
               This ensures the model only learns to predict the response.
            9. Create `attention_mask` as a tensor of ones with the same length.
            10. Return the dict with all three tensors.
        """
        raise NotImplementedError("TODO: implement __getitem__")


def format_instruction_exercise(
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

    Hints:
        - For "alpaca" template, use sections "### Instruction:", "### Input:",
          "### Response:" separated by double newlines.
        - Only include the "### Input:" section if `input_text` is non-empty.
        - For "simple" template, use "Instruction:", "Input:", "Output:"
          separated by single newlines.
        - The output is always appended after the response/output marker.
    """
    raise NotImplementedError("TODO: implement format_instruction")


def compute_instruction_loss_exercise(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute loss for instruction tuning.

    Only computes loss on the response portion (labels != -100).

    Args:
        model: The language model.
        batch: Dict with "input_ids" and "labels".

    Returns:
        Scalar loss tensor.

    Hints:
        1. Extract `input_ids` and `labels` from the batch.
        2. Call `model(input_ids)` to get outputs.
        3. The output may be a tensor or an object with a `.logits` attribute.
           Use `isinstance(outputs, torch.Tensor)` to check.
        4. Compute cross-entropy loss with `ignore_index=-100` so that
           masked positions (the instruction) do not contribute to the loss.
        5. Reshape logits to (batch*seq, vocab) and labels to (batch*seq)
           before computing loss.
    """
    raise NotImplementedError("TODO: implement compute_instruction_loss")
