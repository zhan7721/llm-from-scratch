# Data Engineering for LLM Pretraining

Data quality is arguably the single most important factor in training capable language models. As the saying goes: garbage in, garbage out. This chapter covers three core data engineering techniques used in modern LLM pretraining pipelines.

## Why Data Quality Matters

The data used to pretrain an LLM shapes everything the model learns. A model trained on noisy, duplicated, or low-quality text will produce noisy, low-quality outputs. Key findings from recent research:

- **GPT-3** (2020): Careful filtering of Common Crawl data was essential. They trained a classifier to distinguish "high-quality" web text from low-quality pages.
- **The Pile** (2020): EleutherAI showed that a carefully curated mixture of 22 diverse datasets produces better models than raw web scrapes alone.
- **Chinchilla** (2022): DeepMind demonstrated that data quality matters more than sheer quantity -- a smaller model trained on better data can outperform a larger model trained on worse data.
- **FineWeb** (2024): HuggingFace showed that aggressive quality filtering of Common Crawl yields a dataset that outperforms all prior open web datasets.

### The Data Pipeline

A typical pretraining data pipeline follows these stages:

```
Raw crawl data
    |
    v
URL filtering         -- remove known bad domains
    |
    v
Text extraction       -- HTML to clean text
    |
    v
Language detection    -- keep target language(s)
    |
    v
Deduplication         -- remove exact and near-duplicates
    |
    v
Quality filtering     -- remove low-quality documents
    |
    v
Data mixing           -- blend sources at desired ratios
    |
    v
Tokenization          -- convert text to token sequences
```

This chapter implements the deduplication, quality filtering, and data mixing stages.

## MinHash + LSH: Near-Duplicate Detection

### The Problem

Exact duplicate detection is easy -- just hash each document. But near-duplicates are common and harmful:

- The same Wikipedia article scraped from different mirrors with minor formatting differences
- News articles syndicated across outlets with slight edits
- Boilerplate text repeated across thousands of pages (privacy policies, terms of service)

Training on near-duplicates wastes compute and biases the model toward repeated patterns.

### Shingling

To compare documents, we first convert them to sets of **shingles** -- overlapping character n-grams. For example, with n=5:

```
"hello world" -> {"hello", "ello ", "llo w", "lo wo", "o wor", " worl", "world"}
```

The Jaccard similarity between two documents' shingle sets measures their overlap:

```
J(A, B) = |A intersection B| / |A union B|
```

A Jaccard similarity of 1.0 means identical shingle sets; 0.0 means no overlap.

### MinHash Signatures

Computing Jaccard similarity directly is expensive for large shingle sets. MinHash provides an efficient approximation:

1. Define `k` random hash functions `h_1, h_2, ..., h_k`
2. For each hash function, compute the minimum hash value across all shingles in the document
3. The resulting `k` values form the document's **MinHash signature**

The key insight: for two sets A and B, `P(h_min(A) == h_min(B)) = J(A, B)`. So the fraction of matching positions in two signatures estimates their Jaccard similarity.

### Locality-Sensitive Hashing (LSH)

Even with MinHash, pairwise comparison of all signatures is O(n^2). LSH accelerates this by:

1. Dividing each signature into `b` bands of `r` rows
2. Hashing each band to a bucket
3. Only comparing documents that share at least one bucket

This turns the problem from O(n^2) to approximately O(n), with tunable sensitivity via the band/row ratio.

### Implementation Walkthrough

```python
class MinHashDeduplicator:
    def __init__(self, num_hashes=128, ngram_size=5, threshold=0.8):
        ...

    def _get_shingles(self, text):
        # Extract character n-grams
        text = text.lower().strip()
        return {text[i:i+self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _minhash(self, shingles):
        # For each hash function, find the minimum hash across all shingles
        signature = []
        for a, b in self.hash_params:
            min_hash = float('inf')
            for shingle in shingles:
                h = (a * hash(shingle) + b) & 0xFFFFFFFF
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def _estimate_similarity(self, sig1, sig2):
        # Fraction of matching positions estimates Jaccard similarity
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def deduplicate(self, documents):
        # Compute signatures, then pairwise compare
        signatures = [self._minhash(self._get_shingles(doc)) for doc in documents]
        keep = [True] * len(documents)
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                if self._estimate_similarity(signatures[i], signatures[j]) >= self.threshold:
                    keep[j] = False
        return [doc for doc, k in zip(documents, keep) if k]
```

## Quality Filtering

Not all web text is suitable for training. Quality filters remove documents that are likely to degrade model performance.

### Common Quality Signals

| Signal | Why It Matters |
|--------|---------------|
| Word count | Too few words = not informative; too many = likely a data dump |
| Alpha ratio | Low ratio means lots of numbers, code, or garbled text |
| Digit ratio | High ratio means numerical data, tables, or spam |
| Average word length | Very short = possibly not real text; very long = possibly URLs or hashes |

### Advanced Filters (Not Implemented Here)

Production pipelines use additional filters:

- **Perplexity filtering**: Use a small LM to score text; reject very low perplexity (repetitive) or very high perplexity (garbled) documents
- **Classifier-based filtering**: Train a classifier to distinguish Wikipedia-like text from random web text (GPT-3 approach)
- **PII detection**: Remove documents containing emails, phone numbers, SSNs
- **Toxicity filtering**: Remove hateful, offensive, or harmful content
- **Heuristic rules**: Excessive capitalization, too many special characters, repeated n-grams

### Implementation Walkthrough

```python
class QualityFilter:
    def _is_quality(self, text):
        words = text.split()
        n_words = len(words)

        # Check word count bounds
        if n_words < self.min_words or n_words > self.max_words:
            return False

        # Check that enough characters are alphabetic
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / max(len(text), 1) < self.min_alpha_ratio:
            return False

        # Check that not too many characters are digits
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count / max(len(text), 1) > self.max_digit_ratio:
            return False

        # Check average word length
        avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
        if not (self.min_avg_word_len <= avg_word_len <= self.max_avg_word_len):
            return False

        return True
```

## Data Mixing

### Why Mix Sources?

LLMs benefit from diverse training data. A model trained only on Wikipedia will be knowledgeable but rigid; one trained only on web text will be fluent but unreliable. The key is finding the right mixture.

### Domain Ratios

Common mixing strategies:

- **Proportional**: Sample from each source proportional to its size (large web corpora dominate)
- **Upsampling**: Over-represent high-quality sources (Wikipedia, books) relative to their natural frequency
- **Temperature-based**: Use temperature scaling to smooth or sharpen the distribution

GPT-3's mixture (simplified):
- 60% filtered Common Crawl
- 22% WebText2
- 16% Books (Books1 + Books2)
- 3% Wikipedia

### Curriculum Learning

Some research suggests ordering data by quality or difficulty:

1. Start with cleaner, simpler text
2. Gradually introduce more complex or noisy data
3. End with the highest-quality data

This can improve convergence speed and final performance, though the evidence is mixed.

### Implementation Walkthrough

```python
class DataMixer:
    def __init__(self, ratios):
        # Normalize ratios to sum to 1.0
        total = sum(ratios.values())
        self.ratios = {k: v / total for k, v in ratios.items()}

    def mix(self, data, total_tokens=None, tokens_per_doc=512):
        if total_tokens is None:
            # Proportional mixing from available data
            min_docs = min(len(docs) for docs in data.values())
            result = []
            for source, ratio in self.ratios.items():
                n = max(1, int(min_docs * ratio))
                result.extend(data[source][:n])
            return result

        # Token-budget mixing
        result = []
        for source, ratio in self.ratios.items():
            n_docs = max(1, int((total_tokens * ratio) / tokens_per_doc))
            result.extend(data[source][:n_docs])
        return result
```

## Real-World Data Pipelines

### The Pile (EleutherAI, 2020)

A 825 GiB English text corpus designed for LLM training, composed of 22 diverse sub-corpora:

- Pile-CC (filtered Common Crawl)
- PubMed, ArXiv, GitHub, StackExchange
- Wikipedia, BookCorpus2, Project Gutenberg
- And many more

Key innovation: diversity through intentional source mixing rather than relying on a single massive crawl.

### RedPajama (Together AI, 2023)

An open reproduction of the LLaMA training data recipe:

- 1.2 trillion tokens
- Sources: Common Crawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange
- Documented filtering and deduplication pipeline

### FineWeb (HuggingFace, 2024)

A 15 trillion token dataset from Common Crawl with extensive filtering:

- URL filtering (remove known low-quality domains)
- Text extraction with trafilatura
- Language identification with fastText
- MinHash deduplication
- Perplexity filtering with a small LM
- Repetition and quality heuristics

FineWeb-Edu further filters for "educational" content using a classifier trained on educational web pages.

## Running the Code

```bash
# Run the tests
pytest tests.py -v

# Try the exercises
# Edit exercise.py to implement the TODO sections, then run tests
pytest tests.py -v
```

## Key Takeaways

1. **Deduplication is essential**: Near-duplicates waste compute and bias training. MinHash provides an efficient approximation for large-scale dedup.

2. **Quality filtering improves results**: Simple heuristics (word count, character ratios) can dramatically improve dataset quality. Production pipelines use classifier-based filtering.

3. **Mixing ratios matter**: The proportion of different data sources significantly affects model capabilities. High-quality sources should be upsampled relative to their natural frequency.

4. **The pipeline is iterative**: Data engineering is not a one-shot process. Expect to iterate on filters, thresholds, and ratios based on downstream model performance.
