"""Data engineering tools for pretraining: deduplication, quality filtering, data mixing."""

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
        """Compute MinHash signature for a set of shingles."""
        signature = []
        for a, b in self.hash_params:
            min_hash = float('inf')
            for shingle in shingles:
                h = (a * hash(shingle) + b) & 0xFFFFFFFF
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def _estimate_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def deduplicate(self, documents: List[str]) -> List[str]:
        """Remove near-duplicate documents.

        Args:
            documents: List of text documents.

        Returns:
            De-duplicated list of documents.
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
        """Check if text passes quality thresholds."""
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
        """Filter documents by quality criteria.

        Args:
            documents: List of text documents.

        Returns:
            List of documents that pass quality checks.
        """
        return [doc for doc in documents if self._is_quality(doc)]


class DataMixer:
    """Mix data from multiple sources with configurable ratios.

    Example:
        mixer = DataMixer({"wiki": 0.6, "books": 0.3, "code": 0.1})
        mixed = mixer.mix({"wiki": wiki_data, "books": books_data, "code": code_data}, total_tokens=1000000)
    """

    def __init__(self, ratios: Dict[str, float]):
        """
        Args:
            ratios: Dict mapping source name to its proportion (should sum to ~1.0).
        """
        total = sum(ratios.values())
        self.ratios = {k: v / total for k, v in ratios.items()}

    def mix(
        self,
        data: Dict[str, List[str]],
        total_tokens: Optional[int] = None,
        tokens_per_doc: int = 512,
    ) -> List[str]:
        """Mix data from multiple sources according to configured ratios.

        Args:
            data: Dict mapping source name to list of documents.
            total_tokens: Total number of tokens desired (approximate).
            tokens_per_doc: Estimated tokens per document for sizing.

        Returns:
            Mixed list of documents.
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
