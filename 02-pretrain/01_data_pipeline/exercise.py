"""Data Pipeline for LLM Pretraining -- Exercise.

Fill in the TODO methods to implement the data pipeline components.
Start with `PretrainDataset.__getitem__`, then `dynamic_pad_collate`,
and finally `PackedDataset.__getitem__`.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict


class PretrainDatasetExercise(Dataset):
    """Dataset that returns contiguous token sequences for pretraining.

    Splits a flat token array into fixed-length chunks.
    """

    def __init__(self, token_ids: List[int], seq_len: int):
        self.seq_len = seq_len
        # Truncate to fit complete sequences
        n_tokens = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n_tokens], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) // self.seq_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single training sample.

        Args:
            idx: Index of the chunk to retrieve.

        Returns:
            A dict with:
                - "input_ids": tensor of shape (seq_len,), the input tokens.
                - "labels": tensor of shape (seq_len,), the target tokens
                  shifted by one position to the right.

        Hints:
            - Compute `start = idx * self.seq_len`.
            - Extract a chunk of length `seq_len + 1` from self.data.
            - "input_ids" is everything except the last token.
            - "labels" is everything except the first token.
            - This shift means the model learns to predict the *next* token.
        """
        raise NotImplementedError("TODO: implement __getitem__")


def dynamic_pad_collate_exercise(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function with dynamic padding.

    Pads sequences to the max length in the batch (not a fixed length).

    Args:
        batch: A list of dicts, each with "input_ids" and "labels" tensors
               of potentially different lengths.
        pad_id: The token ID to use for padding.

    Returns:
        A dict with:
            - "input_ids": (batch_size, max_len) tensor, padded with pad_id.
            - "labels": (batch_size, max_len) tensor, padded with -100
              (the ignore index for PyTorch's CrossEntropyLoss).
            - "attention_mask": (batch_size, max_len) tensor, 1 for real
              tokens and 0 for padded positions.

    Hints:
            - Find the max length across all items in the batch.
            - For each item, compute how much padding is needed.
            - Use `torch.cat` to append padding tokens.
            - Use `torch.full` to create padding tensors.
            - Use `torch.stack` to combine all items into a single tensor.
            - Labels use -100 for padding so the loss function ignores them.
    """
    raise NotImplementedError("TODO: implement dynamic_pad_collate")


class PackedDatasetExercise(Dataset):
    """Packs multiple sequences into fixed-length blocks to eliminate padding waste.

    Concatenates all tokens, then splits into blocks of seq_len + 1.
    """

    def __init__(self, token_ids_list: List[List[int]], seq_len: int, pad_id: int = 0):
        self.seq_len = seq_len
        # Concatenate with separator between documents
        all_tokens = []
        for doc_tokens in token_ids_list:
            all_tokens.extend(doc_tokens)
            all_tokens.append(pad_id)  # separator

        # Truncate to fit complete blocks
        n_tokens = (len(all_tokens) // (seq_len + 1)) * (seq_len + 1)
        self.data = torch.tensor(all_tokens[:n_tokens], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single packed training sample.

        Args:
            idx: Index of the block to retrieve.

        Returns:
            A dict with:
                - "input_ids": tensor of shape (seq_len,), the input tokens.
                - "labels": tensor of shape (seq_len,), the target tokens
                  shifted by one position to the right.

        Hints:
            - Compute `start = idx * (self.seq_len + 1)`.
            - Extract a chunk of length `seq_len + 1`.
            - "input_ids" is everything except the last token.
            - "labels" is everything except the first token.
            - This is the same shifting logic as PretrainDataset.
        """
        raise NotImplementedError("TODO: implement __getitem__")
