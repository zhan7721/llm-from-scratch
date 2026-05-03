"""Data engineering solution: complete reference implementation.

This file contains the full solutions for the exercises in exercise.py.
"""

import hashlib
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math


class MinHashDeduplicator:
    """Near-duplicate detection using MinHash + LSH.

    Uses character n-gram shingles and MinHash signatures to find
    documents that are likely near-duplicates.
    """

    def __init__(self, num_hashes: int = 128, ngram_size: int = 5, threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.threshold = threshold
        # Generate random hash functions (seed-based for reproducibility)
        self.hash_params = [
            (i * 2654435761 & 0xFFFFFFFF, i * 2246822519 & 0xFFFFFFFF)
            for i in range(num_hashes)
        ]

    def _get_shingles(self, text: str) -> set:
        """Extract character n-gram shingles from text."""
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return {text}
        return {text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _minhash(self, shingles: set) -> List[int]:
        """Compute MinHash signature for a set of shingles.

        For each hash function (parameterized by a, b from self.hash_params):
        1. For every shingle in the set, compute h = (a * hash(shingle) + b) & 0xFFFFFFFF
        2. Take the minimum hash value across all shingles
        3. Append that minimum to the signature
        """
        signature = []
        for a, b in self.hash_params:
            min_hash = float('inf')
            for shingle in shingles:
                h = (a * hash(shingle) + b) & 0xFFFFFFFF
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def _estimate_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures.

        The fraction of matching positions in two signatures estimates
        the Jaccard similarity of the original sets.
        """
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def deduplicate(self, documents: List[str]) -> List[str]:
        """Remove near-duplicate documents.

        Steps:
        1. Compute MinHash signature for each document
        2. Compare all pairs of signatures
        3. If similarity >= threshold, mark the later document as duplicate
        4. Return only non-duplicate documents
        """
        if not documents:
            return []

        # Compute signatures
        signatures = []
        for doc in documents:
            shingles = self._get_shingles(doc)
            sig = self._minhash(shingles)
            signatures.append(sig)

        # Find duplicates via pairwise comparison
        keep = [True] * len(documents)
        for i in range(len(documents)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(documents)):
                if not keep[j]:
                    continue
                sim = self._estimate_similarity(signatures[i], signatures[j])
                if sim >= self.threshold:
                    keep[j] = False

        return [doc for doc, k in zip(documents, keep) if k]


class QualityFilter:
    """Rule-based text quality filtering for pretraining data."""

    def __init__(
        self,
        min_words: int = 10,
        max_words: int = 100000,
        min_alpha_ratio: float = 0.5,
        max_digit_ratio: float = 0.5,
        min_avg_word_len: float = 2.0,
        max_avg_word_len: float = 20.0,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_alpha_ratio = min_alpha_ratio
        self.max_digit_ratio = max_digit_ratio
        self.min_avg_word_len = min_avg_word_len
        self.max_avg_word_len = max_avg_word_len

    def _is_quality(self, text: str) -> bool:
        """Check if text passes quality thresholds.

        Checks:
        1. Word count is between min_words and max_words
        2. Ratio of alphabetic characters >= min_alpha_ratio
        3. Ratio of digit characters <= max_digit_ratio
        4. Average word length is between min_avg_word_len and max_avg_word_len
        """
        words = text.split()
        n_words = len(words)

        if n_words < self.min_words or n_words > self.max_words:
            return False

        # Alpha character ratio
        alpha_count = sum(1 for c in text if c.isalpha())
        total_count = max(len(text), 1)
        if alpha_count / total_count < self.min_alpha_ratio:
            return False

        # Digit ratio
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count / total_count > self.max_digit_ratio:
            return False

        # Average word length
        avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
        if avg_word_len < self.min_avg_word_len or avg_word_len > self.max_avg_word_len:
            return False

        return True

    def filter(self, documents: List[str]) -> List[str]:
        """Filter documents by quality criteria."""
        return [doc for doc in documents if self._is_quality(doc)]


class DataMixer:
    """Mix data from multiple sources with configurable ratios."""

    def __init__(self, ratios: Dict[str, float]):
        total = sum(ratios.values())
        self.ratios = {k: v / total for k, v in ratios.items()}

    def mix(
        self,
        data: Dict[str, List[str]],
        total_tokens: Optional[int] = None,
        tokens_per_doc: int = 512,
    ) -> List[str]:
        """Mix data from multiple sources according to configured ratios.

        If total_tokens is None:
            - Take the minimum document count across sources
            - For each source, take int(min_docs * ratio) documents (at least 1)

        If total_tokens is provided:
            - For each source, compute n_docs = int((total_tokens * ratio) / tokens_per_doc)
            - Take that many documents from each source (at least 1)
        """
        if total_tokens is None:
            # Just mix proportionally from available data
            min_docs = min(len(docs) for docs in data.values())
            result = []
            for source, ratio in self.ratios.items():
                n = max(1, int(min_docs * ratio))
                result.extend(data[source][:n])
            return result

        result = []
        for source, ratio in self.ratios.items():
            n_docs = max(1, int((total_tokens * ratio) / tokens_per_doc))
            available = data.get(source, [])
            result.extend(available[:n_docs])

        return result
