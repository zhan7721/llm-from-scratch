# Data Pipeline

> **Module 02 -- Pretraining, Chapter 01**

Before a language model can learn anything, it needs a steady stream of well-structured training data. The data pipeline is the plumbing that turns raw token IDs into batches of tensors ready for the GPU. A poorly designed pipeline wastes compute on padding, shuffles data incorrectly, or produces batches with mismatched shapes. This chapter builds the pipeline from scratch.

---

## Prerequisites

- Basic PyTorch (`Dataset`, `DataLoader`, `torch.tensor`)
- Understanding of tokenization (Module 01, Chapter 01)

## Files

| File | Purpose |
|------|---------|
| `data_pipeline.py` | Core implementation: datasets, collation, packing |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |
| `test_exercise.py` | pytest tests for the exercise version |

---

## Why the Data Pipeline Matters

Pretraining an LLM means processing billions of tokens. Even a small inefficiency -- like padding every sequence to a fixed 2048 tokens when the average length is 300 -- multiplies across billions of steps and wastes enormous amounts of compute. The data pipeline must:

1. **Split tokens into fixed-length chunks** so every sample has the same shape.
2. **Create input/target pairs** with a one-token shift (the model learns to predict the next token).
3. **Batch samples efficiently** without wasting GPU cycles on padding.
4. **Shuffle data** so the model does not memorize the order of the training corpus.

---

## Dataset Design: Contiguous Sequences

The simplest approach is `PretrainDataset`. It takes a flat list of token IDs and splits it into non-overlapping chunks of length `seq_len`.

```
Tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
                         seq_len = 4

Chunk 0: input = [0, 1, 2, 3],  label = [1, 2, 3, 4]
Chunk 1: input = [4, 5, 6, 7],  label = [5, 6, 7, 8]
Chunk 2: input = [8, 9, 10, 11], label = [9, 10, 11, 12]
```

Key details:

- **Truncation**: We drop any trailing tokens that do not fill a complete chunk. This keeps all samples the same length.
- **Input/target shift**: `labels` is `input_ids` shifted one position to the right. This is the standard autoregressive language modeling objective -- given tokens up to position `t`, predict the token at position `t+1`.
- **No overlap**: Chunks are contiguous and non-overlapping. This is the simplest strategy. Overlapping (sliding window) is also possible but doubles memory usage.

```python
class PretrainDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        n_tokens = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n_tokens], dtype=torch.long)

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }
```

---

## Dynamic Padding

When sequences in a batch have different lengths, we must pad them to the same length so they can be stacked into a single tensor. There are two strategies:

### Fixed padding (wasteful)

Pad every sequence to a global maximum (e.g., 2048). If most sequences are 300 tokens, you waste 85% of your compute on padding.

### Dynamic padding (efficient)

Pad each batch to the length of the longest sequence *in that batch*. If a batch has sequences of lengths [300, 280, 310, 295], you pad to 310 instead of 2048.

```python
def dynamic_pad_collate(batch, pad_id=0):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    # ... pad each sequence to max_len ...
```

The collate function also produces:

- **`attention_mask`**: A binary tensor (1 for real tokens, 0 for padding). The model uses this to avoid attending to padding positions.
- **`labels` with -100**: PyTorch's `CrossEntropyLoss` ignores positions with label -100. We set padded positions to -100 so the loss is only computed on real tokens.

---

## Packing: Eliminating Padding Waste Entirely

Dynamic padding helps, but even within a batch, shorter sequences still waste some space. **Packing** takes this further by eliminating padding completely.

The idea: concatenate all documents into one long token stream, then split into fixed-length blocks.

```
Document A: [a1, a2, a3, a4, a5]
Document B: [b1, b2, b3]

Packed stream: [a1, a2, a3, a4, a5, SEP, b1, b2, b3, ...]

Block 0 (seq_len=4): input = [a1, a2, a3, a4], label = [a2, a3, a4, a5]
Block 1 (seq_len=4): input = [a5, SEP, b1, b2], label = [SEP, b1, b2, b3]
```

### Document boundaries

A subtle problem: if document A ends at position 3 and document B starts at position 4 in the same block, the attention mechanism will attend across the boundary. The model sees tokens from document B when predicting the last token of document A.

Solutions:

1. **Separator tokens**: Insert a special token between documents. The model learns that after a separator, the context resets.
2. **Attention mask modification**: Use a block-diagonal attention mask that prevents cross-document attention. This is more complex but more principled.
3. **Accept the leakage**: In practice, for large-scale pretraining, the small amount of cross-document attention at boundaries has minimal impact.

Our implementation uses approach 1 (separator tokens) for simplicity.

---

## DataLoader Configuration

The `create_pretrain_dataloader` function wraps everything together:

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,         # randomize sample order each epoch
    collate_fn=...,       # dynamic padding (or None for packing)
    drop_last=True,       # drop the last incomplete batch
)
```

### `shuffle=True`

Pretraining data is a flat stream of tokens. Shuffling prevents the model from learning spurious patterns in the order of documents. Each epoch sees the same data in a different order.

### `drop_last=True`

The last batch may be smaller than `batch_size`. Dropping it avoids:
- Wasting compute on a small batch that cannot saturate the GPU.
- Potential shape issues with batch normalization (not used in transformers, but a good habit).

### `num_workers`

For production training, set `num_workers > 0` to load data in parallel processes. This keeps the GPU fed while the CPU prepares the next batch. For learning and debugging, `num_workers=0` (the default) is fine.

---

## Code Walkthrough

### Step 1: PretrainDataset

```python
dataset = PretrainDataset(token_ids, seq_len=512)
```

Takes a flat list of token IDs (e.g., the entire training corpus tokenized) and splits it into chunks of 512 tokens. Each `__getitem__` call returns a dict with `input_ids` (512 tokens) and `labels` (the same 512 tokens shifted by one).

### Step 2: dynamic_pad_collate

```python
collate_fn = lambda b: dynamic_pad_collate(b, pad_id=0)
```

When the DataLoader collects a batch of samples, this function pads them to the same length. It also creates the attention mask and sets label padding to -100.

### Step 3: PackedDataset

```python
dataset = PackedDataset([doc1_tokens, doc2_tokens, ...], seq_len=512)
```

Takes multiple documents (each a list of token IDs), concatenates them with separators, and splits into blocks. No padding needed -- every block is exactly `seq_len` tokens.

### Step 4: create_pretrain_dataloader

```python
loader = create_pretrain_dataloader(
    token_ids,
    seq_len=512,
    batch_size=8,
    use_packing=False,
)
```

One function to rule them all. Creates either a `PretrainDataset` or `PackedDataset` depending on `use_packing`, wraps it in a DataLoader with the appropriate collate function, and returns a ready-to-iterate training loader.

---

## Exercises

Open `exercise.py` and implement:

1. **`PretrainDataset.__getitem__`**: Extract a chunk and create the input/label shift.
2. **`dynamic_pad_collate`**: Pad a batch to the max length, create attention masks, set label padding to -100.
3. **`PackedDataset.__getitem__`**: Same as PretrainDataset but using the packed block size.

Run the exercise tests with:

```bash
pytest test_exercise.py -v
```

Remove the `@pytest.mark.skip` decorators as you implement each method.

---

## Running Tests

```bash
# Test the main implementation
pytest tests.py -v

# Test the exercise version
pytest test_exercise.py -v
```

---

## Summary

| Concept | What it solves |
|---------|---------------|
| `PretrainDataset` | Splits flat tokens into fixed-length input/label pairs |
| `dynamic_pad_collate` | Pads batches to the max length in the batch, not a global max |
| `PackedDataset` | Eliminates padding entirely by concatenating documents |
| `attention_mask` | Tells the model which positions are real tokens vs. padding |
| `labels = -100` | Tells the loss function to ignore padded positions |
| `drop_last=True` | Avoids wasting compute on a small final batch |

---

## Next Steps

- **Chapter 02 (Data Engineering)**: Where the raw text comes from, cleaning, deduplication, and quality filtering.
- **Chapter 03 (Training Loop)**: How to actually train a model using this data pipeline.
