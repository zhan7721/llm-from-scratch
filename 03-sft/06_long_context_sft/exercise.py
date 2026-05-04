"""Long Context SFT -- Exercise.

Fill in the TODO methods to implement long-context fine-tuning components.
Start with `PositionInterpolation.rescale_positions`, then `NTKAwareScaling.get_scaled_inv_freq`,
`LongContextDataset.__getitem__`, and finally `compute_long_context_loss`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple


class PositionInterpolationExercise:
    """Position Interpolation (PI) for extending context length.

    Linearly scales position indices to fit within the original
    training context window.
    """

    def __init__(self, original_max_seq_len: int, target_max_seq_len: int):
        self.original_max_seq_len = original_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.scale_factor = target_max_seq_len / original_max_seq_len

    def rescale_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Rescale positions from target range to original range.

        Args:
            positions: Position indices in [0, target_max_seq_len).

        Returns:
            Rescaled positions in [0, original_max_seq_len).

        Hints:
            1. The scale factor is target_max_seq_len / original_max_seq_len.
            2. To map positions back into the original range, divide by the scale factor.
            3. Example: if original=2048, target=8192, scale=4.
               Position 8191 becomes 8191/4 = 2047.75.
        """
        raise NotImplementedError("TODO: implement rescale_positions")


class NTKAwareScalingExercise:
    """NTK-aware scaling for RoPE context extension.

    Adjusts the base frequency of RoPE to preserve high-frequency
    components while extending context.
    """

    def __init__(self, original_max_seq_len: int, target_max_seq_len: int, base: float = 10000.0):
        self.original_max_seq_len = original_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.scale_factor = target_max_seq_len / original_max_seq_len

        # TODO: Compute the scaled base.
        # NTK-aware: scale the base frequency instead of the positions.
        # The formula: scaled_base = base * (scale_factor ^ (2*pi / (2*pi)))
        # Which simplifies to: scaled_base = base * scale_factor
        # But the general form preserves the structure for other exponents.
        self.scaled_base = None  # TODO: compute this

    def get_scaled_inv_freq(self, d_model: int, device: torch.device) -> torch.Tensor:
        """Get scaled inverse frequency for RoPE.

        Args:
            d_model: Model dimension (must be even).
            device: Torch device.

        Returns:
            Inverse frequency tensor of shape (d_model // 2,).

        Hints:
            1. Use self.scaled_base instead of the original base.
            2. Generate indices: arange(0, d_model, 2) -> shape (d_model//2,).
            3. Formula: inv_freq = 1.0 / (scaled_base ** (indices / d_model))
        """
        raise NotImplementedError("TODO: implement get_scaled_inv_freq")


class LongContextDatasetExercise(torch.utils.data.Dataset):
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
                # Short document: pad to seq_len + 1 (for input/label shift)
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
        """Get a single training example.

        Args:
            idx: Index into the chunks list.

        Returns:
            Dict with "input_ids" and "labels".

        Hints:
            1. Get the chunk at index `idx` from self.chunks.
            2. input_ids = chunk[:-1] (all tokens except the last).
            3. labels = chunk[1:] (all tokens except the first, shifted by one).
            4. Convert both to torch.long tensors.
            5. Return as a dict with keys "input_ids" and "labels".
        """
        raise NotImplementedError("TODO: implement __getitem__")


def compute_long_context_loss_exercise(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute loss for long sequences by chunking.

    Processes long sequences in chunks to avoid OOM,
    then aggregates the loss.

    Args:
        model: Language model.
        batch: Dict with input_ids and labels.
        chunk_size: Size of each chunk for processing.

    Returns:
        Average loss across chunks.

    Hints:
        1. Extract input_ids and labels from the batch.
        2. Get seq_len from input_ids.shape[1].
        3. If seq_len <= chunk_size, compute loss normally (no chunking needed).
        4. Otherwise, loop over the sequence in chunk_size steps:
           - Slice input_ids[:, start:end] and labels[:, start:end].
           - Compute logits = model(chunk_input).
           - Handle the case where outputs is a tensor or has a .logits attribute.
           - Compute cross_entropy loss for this chunk.
           - Accumulate total_loss and count chunks.
        5. Return total_loss / n_chunks (average loss).
    """
    raise NotImplementedError("TODO: implement compute_long_context_loss")
