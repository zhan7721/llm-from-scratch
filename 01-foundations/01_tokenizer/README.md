# BPE Tokenizer

> **Module 01 — Foundations, Chapter 01**

Large Language Models do not read text directly. They operate on numbers. A **tokenizer** is the bridge between human language and the model's numerical world. In this chapter we build a [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer from scratch — the same fundamental algorithm used by GPT-2, LLaMA, and virtually every modern LLM.

---

## Prerequisites

- Basic Python (dicts, tuples, loops)
- No ML or PyTorch required for this chapter

## Files

| File | Purpose |
|------|---------|
| `tokenizer.py` | Core BPE implementation (train, encode, decode) |
| `train_tokenizer.py` | Demo script — trains on sample text and shows results |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |
| `test_exercise.py` | pytest tests for the exercise version |

---

## Why Tokenization?

A neural network is a mathematical function. It takes numbers in and produces numbers out. Raw text like `"Hello world"` is not numbers — it is a sequence of Unicode characters. We need a deterministic, reversible mapping from text to integers.

The simplest approach: assign each character an ID.

```
H → 72, e → 101, l → 108, l → 108, o → 111, ...
```

This works, but it has problems:

1. **Vocabulary explosion.** Unicode has over 149,000 characters. A character-level tokenizer needs a huge vocabulary, and rare characters cause issues.
2. **No semantic grouping.** The word `"playing"` is split into 7 separate tokens. The model must learn from scratch that `p-l-a-y-i-n-g` is a single concept.
3. **Long sequences.** More tokens per word means longer input sequences, which cost more compute.

**BPE** solves these problems by learning **subword units** — chunks that are larger than individual characters but smaller than whole words. Common patterns like `"the"`, `"ing"`, or `"tion"` become single tokens, while rare words are broken into known subwords.

---

## The BPE Algorithm

BPE stands for **Byte Pair Encoding**. It was originally a data compression algorithm ([Gage, 1994](https://www.derczynski.com/papers/archive/bpe_gage.pdf)). The key insight is simple:

> Repeatedly find the most frequent adjacent pair of tokens and merge them into a new token.

### Step-by-Step Example

Let's walk through BPE on a tiny corpus:

```
corpus = "aa ab aa ab aa ab"
```

**Step 0: Initialize**

Split the corpus into words and count frequencies:

```
"aa" → 3 times
"ab" → 3 times
```

Represent each word as a sequence of bytes:

```
"a", "a" → 3
"a", "b" → 3
```

Our initial vocabulary is the 256 possible byte values (0–255).

**Step 1: Count pairs and merge**

Count all adjacent pairs across all words:

```
(a, a) → 3 occurrences  (from "aa")
(a, b) → 3 occurrences  (from "ab")
```

Both pairs have the same frequency (3). Ties are broken arbitrarily. Let's say we pick `(a, a)`.

Merge `(a, a)` into a new token `aa`. It gets ID 256.

```
word_freqs becomes:
  (aa,) → 3      # was (a, a)
  (a, b) → 3     # unchanged
```

**Step 2: Count pairs and merge again**

```
(a, b) → 3 occurrences
```

Merge `(a, b)` into `ab`. It gets ID 257.

```
word_freqs becomes:
  (aa,) → 3
  (ab,) → 3
```

**Step 3: No more pairs**

Every word is now a single token. There are no adjacent pairs to merge. The algorithm stops.

**Result:**

- Vocabulary: 256 base bytes + `aa` (ID 256) + `ab` (ID 257) = 258 tokens
- Merges learned: `['a' + 'a' → 'aa']`, `['a' + 'b' → 'ab']`

The input text `"aa ab aa ab"` can now be encoded as `[256, 257, 256, 257]` — just 4 tokens instead of 11 bytes.

### Why Bytes?

Modern BPE implementations (including ours) operate on **bytes**, not characters. This has a major advantage: every possible string can be represented, even if it contains unusual Unicode, emojis, or binary data. The base vocabulary is always exactly 256 tokens — one for each possible byte value.

---

## Code Walkthrough

Here is how the core `BPETokenizer` class in `tokenizer.py` works.

### Training: `train(corpus)`

```python
def train(self, corpus: str):
    self._build_base_vocab()           # 1. Initialize 256-byte vocab
    words = corpus.split()             # 2. Split into words
    word_freqs = ...                   # 3. Count word frequencies
    # Each word is a tuple of single-byte tokens

    for _ in range(num_merges):        # 4. Repeat (vocab_size - 256) times
        pair_counts = self._get_pair_counts(word_freqs)  # Count pairs
        best_pair = max(pair_counts, key=...)             # Find most frequent
        merged = best_pair[0] + best_pair[1]              # Concatenate bytes
        self.vocab[new_id] = merged    # Add to vocabulary
        self.merges.append(best_pair)  # Record the merge rule
        word_freqs = self._merge_pair(word_freqs, best_pair)  # Apply merge
```

The training loop is the heart of BPE. Each iteration finds the most common pair and merges it. The `word_freqs` dictionary is updated in place, so the next iteration sees the merged tokens.

### Encoding: `encode(text)`

```python
def encode(self, text: str) -> List[int]:
    tokens = [bytes([b]) for b in text.encode("utf-8")]  # Start with bytes
    for pair in self.merges:              # Apply each merge rule in order
        tokens = merge(tokens, pair)
    return [self.inverse_vocab[t] for t in tokens]  # Convert to IDs
```

Encoding applies the learned merge rules **in the same order they were discovered during training**. This greedy left-to-right application is what makes encoding fast and deterministic.

### Decoding: `decode(ids)`

```python
def decode(self, ids: List[int]) -> str:
    token_bytes = b"".join(self.vocab[id] for id in ids)
    return token_bytes.decode("utf-8", errors="replace")
```

Decoding is straightforward: look up each ID in the vocabulary, concatenate the bytes, and decode to a string. The `errors="replace"` flag ensures robustness against invalid byte sequences.

### Key Data Structures

- **`vocab`**: Maps token ID (int) to bytes. Size = 256 + number of merges.
- **`inverse_vocab`**: Reverse lookup — bytes to ID. Used during encoding.
- **`merges`**: Ordered list of `(bytes, bytes)` pairs learned during training. This is the "model" — it defines how to compress text.

---

## How to Run

### Train the Tokenizer

```bash
cd /path/to/llm-from-scratch
python 01-foundations/01_tokenizer/train_tokenizer.py
```

Expected output:

```
Vocabulary size: 300
Number of merges: 44

Original: The quick brown fox
Token IDs: [84, 104, 101, ...]
Decoded: The quick brown fox
Roundtrip OK: True

First 10 merges:
  1. b' ' + b't' -> b' t'
  2. b' ' + b'a' -> b' a'
  ...
```

### Run Tests

```bash
pytest 01-foundations/01_tokenizer/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing BPE yourself. The file contains a `BPETokenizerExercise` class with `TODO` placeholders for each key method.

### Exercise Order

1. **`_get_pair_counts`** — Count adjacent pairs across all words
2. **`_merge_pair`** — Replace all occurrences of a pair with the merged token
3. **`train`** — Wire everything together into the training loop
4. **`encode`** — Apply merge rules to encode new text
5. **`decode`** — Convert token IDs back to text

### Tips

- Start with `_get_pair_counts`. It is the simplest method and builds intuition for how BPE counts frequencies.
- `_merge_pair` is the trickiest. Walk through the word tuple left to right, building a new list.
- Once you have `_get_pair_counts` and `_merge_pair`, the `train` method is mostly wiring them together.
- `encode` reuses the same merge logic as `_merge_pair`, but on a flat list instead of a dict.

### Verify Your Solution

```bash
pytest 01-foundations/01_tokenizer/test_exercise.py -v
```

---

## Key Takeaways

1. **Tokenization is a compression problem.** BPE finds recurring byte patterns in your training corpus and merges them into single tokens. More frequent patterns get shorter representations.

2. **BPE operates on bytes, not characters.** This means it can handle any text — English, Chinese, emojis, or binary — without special rules. The base vocabulary is always 256 tokens.

3. **The merge order matters.** Encoding applies merge rules in the exact order they were learned. Different training corpora produce different merge rules and thus different tokenizations.

4. **Subword tokenization balances vocabulary size and sequence length.** Common words become single tokens; rare words are split into known subwords. This keeps the vocabulary manageable while representing any text.

---

## Further Reading

- [Original BPE paper (Sennrich et al., 2016)](https://arxiv.org/abs/1608.00221) — BPE applied to machine translation
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) — Production-grade tokenization library
- [Let's build the GPT Tokenizer (Karpathy)](https://www.youtube.com/watch?v=zduSFxRajkE) — Video walkthrough of BPE
- [minbpe](https://github.com/karpathy/minbpe) — Minimal BPE implementation by Andrej Karpathy
