"""Long Context SFT -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple


class PositionInterpolationSolution:
    """Position Interpolation (PI) for extending context length.

    Linearly scales position indices to fit within the original
    training context window. From Chen et al. 2023.
    """

    def __init__(self, original_max_seq_len: int, target_max_seq_len: int):
        self.original_max_seq_len = original_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.scale_factor = target_max_seq_len / original_max_seq_len

    def rescale_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Rescale positions from target range to original range."""
        return positions / self.scale_factor


class NTKAwareScalingSolution:
    """NTK-aware scaling for RoPE context extension.

    Adjusts the base frequency of RoPE to preserve high-frequency
    components while extending context. From Reddit/bloc97.
    """

    def __init__(self, original_max_seq_len: int, target_max_seq_len: int, base: float = 10000.0):
        self.original_max_seq_len = original_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.scale_factor = target_max_seq_len / original_max_seq_len

        # NTK-aware: scale the base instead of the positions
        self.scaled_base = base * (self.scale_factor ** (2 * math.pi / (2 * math.pi)))

    def get_scaled_inv_freq(self, d_model: int, device: torch.device) -> torch.Tensor:
        """Get scaled inverse frequency for RoPE."""
        inv_freq = 1.0 / (self.scaled_base ** (torch.arange(0, d_model, 2, device=device).float() / d_model))
        return inv_freq


class LongContextDatasetSolution(torch.utils.data.Dataset):
    """Dataset for long-context fine-tuning.

    Supports packing multiple documents with separator tokens,
    and sliding window for long document processing.
    """

    def __init__(
        self,
        documents: List[List[int]],
        seq_len: int,
        separator_id: int = 0,
        stride: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.separator_id = separator_id
        self.stride = stride or seq_len

        # Process documents into chunks
        self.chunks = []
        for doc in documents:
            if len(doc) <= seq_len:
                # Short document: pad to seq_len + 1
                padded = doc + [separator_id] * (seq_len - len(doc) + 1)
                self.chunks.append(padded[:seq_len + 1])
            else:
                # Long document: sliding window
                for start in range(0, len(doc) - seq_len, self.stride):
                    chunk = doc[start:start + seq_len + 1]
                    if len(chunk) == seq_len + 1:
                        self.chunks.append(chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        chunk = self.chunks[idx]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


def compute_long_context_loss_solution(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute loss for long sequences by chunking.

    Processes long sequences in chunks to avoid OOM,
    then aggregates the loss.
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    seq_len = input_ids.shape[1]

    if seq_len <= chunk_size:
        outputs = model(input_ids)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    # Process in chunks
    total_loss = 0.0
    n_chunks = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_input = input_ids[:, start:end]
        chunk_labels = labels[:, start:end]

        outputs = model(chunk_input)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk_labels.reshape(-1), ignore_index=-100)

        total_loss += loss
        n_chunks += 1

    return total_loss / n_chunks


def prepare_long_context_batch_solution(
    token_ids: List[int],
    seq_len: int,
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Prepare a single batch for long-context training."""
    if len(token_ids) >= seq_len:
        input_ids = token_ids[:seq_len]
        labels = token_ids[1:seq_len + 1]
        attention_mask = [1] * seq_len
    else:
        pad_len = seq_len - len(token_ids)
        input_ids = token_ids + [pad_id] * pad_len
        labels = token_ids[1:] + [pad_id] * (pad_len + 1)
        labels = labels[:seq_len]
        attention_mask = [1] * len(token_ids) + [0] * pad_len

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


class LongContextTrainerSolution:
    """Trainer for long-context fine-tuning.

    Handles position interpolation, gradient accumulation,
    and long sequence processing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        original_max_seq_len: int = 2048,
        target_max_seq_len: int = 8192,
        scaling_method: str = "pi",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        if scaling_method == "pi":
            self.scaler = PositionInterpolationSolution(original_max_seq_len, target_max_seq_len)
        else:
            self.scaler = NTKAwareScalingSolution(original_max_seq_len, target_max_seq_len)

    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = self.model(input_ids)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
        }
