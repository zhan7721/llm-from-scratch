"""Byte-Pair Encoding (BPE) Tokenizer implemented from scratch."""

from typing import List, Dict, Tuple


class BPETokenizer:
    """A minimal BPE tokenizer that trains on raw text and encodes/decodes."""

    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self.merges: List[Tuple[bytes, bytes]] = []

    def _build_base_vocab(self):
        """Initialize vocab with all 256 byte values."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inverse_vocab = {bytes([i]): i for i in range(256)}

    def _get_pair_counts(self, word_freqs: Dict[tuple, int]) -> Dict[Tuple[bytes, bytes], int]:
        """Count frequency of adjacent byte pairs across all words."""
        pair_counts: Dict[Tuple[bytes, bytes], int] = {}
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        return pair_counts

    def _merge_pair(self, word_freqs: Dict[tuple, int], pair: Tuple[bytes, bytes]) -> Dict[tuple, int]:
        """Merge all occurrences of the given pair in word_freqs."""
        new_word_freqs = {}
        for word_tuple, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                    new_word.append(word_tuple[i] + word_tuple[i + 1])
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, corpus: str):
        """Train BPE on a text corpus."""
        self._build_base_vocab()
        words = corpus.split()
        word_freqs: Dict[tuple, int] = {}
        for word in words:
            word_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_freqs[word_tuple] = word_freqs.get(word_tuple, 0) + 1

        num_merges = self.vocab_size - 256
        for _ in range(num_merges):
            if not word_freqs:
                break
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            merged = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_id] = merged
            self.inverse_vocab[merged] = new_id
            self.merges.append(best_pair)
            word_freqs = self._merge_pair(word_freqs, best_pair)

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.inverse_vocab[t] for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        token_bytes = b"".join(self.vocab[id] for id in ids)
        return token_bytes.decode("utf-8", errors="replace")
