"""Data Pipeline for LLM Pretraining -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict


class PretrainDatasetSolution(Dataset):
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
        """Return a single training sample."""
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]  # +1 for target
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def dynamic_pad_collate_solution(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function with dynamic padding.

    Pads sequences to the max length in the batch (not a fixed length).
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq = item["input_ids"]
        lab = item["labels"]
        pad_len = max_len - seq.shape[0]

        input_ids.append(torch.cat([seq, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))  # -100 ignored by CE loss
        attention_mask.append(torch.cat([torch.ones(seq.shape[0], dtype=torch.long),
                                         torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


class PackedDatasetSolution(Dataset):
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
        """Return a single packed training sample."""
        start = idx * (self.seq_len + 1)
        chunk = self.data[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def create_pretrain_dataloader_solution(
    token_ids: List[int],
    seq_len: int = 512,
    batch_size: int = 8,
    use_packing: bool = False,
    pad_id: int = 0,
) -> DataLoader:
    """Create a DataLoader for pretraining.

    Args:
        token_ids: Flat list of token IDs.
        seq_len: Sequence length per sample.
        batch_size: Batch size.
        use_packing: If True, use packed dataset (no padding waste).
        pad_id: Padding token ID.

    Returns:
        DataLoader ready for training.
    """
    if use_packing:
        dataset = PackedDatasetSolution([token_ids], seq_len, pad_id)
    else:
        dataset = PretrainDatasetSolution(token_ids, seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None if use_packing else lambda b: dynamic_pad_collate_solution(b, pad_id),
        drop_last=True,
    )
