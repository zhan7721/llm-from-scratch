"""Byte-Pair Encoding (BPE) Tokenizer — Exercise.

Fill in the TODO methods to implement a working BPE tokenizer.
Start with `_get_pair_counts`, then `_merge_pair`, then `train`,
and finally `encode` and `decode`.
"""

from typing import List, Dict, Tuple


class BPETokenizerExercise:
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
        """Count frequency of adjacent byte pairs across all words.

        Args:
            word_freqs: Mapping from word-tuples (each element is a bytes token)
                        to how many times that word appeared in the corpus.

        Returns:
            A dict mapping each adjacent pair (a, b) to its total frequency
            across all words, weighted by each word's frequency.

        Hints:
            - Iterate over every word in word_freqs.
            - For each word, slide a window of size 2 and count pairs.
            - Remember to multiply by the word's frequency.
        """
        raise NotImplementedError("TODO: implement _get_pair_counts")

    def _merge_pair(self, word_freqs: Dict[tuple, int], pair: Tuple[bytes, bytes]) -> Dict[tuple, int]:
        """Merge all occurrences of the given pair in word_freqs.

        Args:
            word_freqs: Current mapping from word-tuples to frequencies.
            pair: The (bytes, bytes) pair to merge wherever it appears.

        Returns:
            A new dict with the same frequencies but with every occurrence
            of `pair` replaced by the single concatenated bytes object.

        Hints:
            - Walk through each word tuple left to right.
            - When you see `pair` at position (i, i+1), concatenate them
              into one bytes object and skip ahead by 2.
            - Otherwise, keep the token and advance by 1.
            - Build a new tuple and add it to the result dict.
        """
        raise NotImplementedError("TODO: implement _merge_pair")

    def train(self, corpus: str):
        """Train BPE on a text corpus.

        Steps:
            1. Call `_build_base_vocab()` to initialize the 256-byte vocabulary.
            2. Split the corpus on whitespace into words.
            3. Convert each word into a tuple of single-byte tokens
               (use `word.encode("utf-8")` and wrap each byte).
            4. Count word frequencies.
            5. Repeat `vocab_size - 256` times:
               a. Get pair counts via `_get_pair_counts`.
               b. If no pairs remain, stop early.
               c. Find the most frequent pair.
               d. Merge that pair into a new token and assign it the next ID.
               e. Record the merge in `self.merges`.
               f. Apply the merge via `_merge_pair`.

        Args:
            corpus: The training text.

        Hints:
            - The merged token is just `pair[0] + pair[1]` (bytes concatenation).
            - New token IDs start at 256 and go up by 1 each merge.
            - You can use `max(pair_counts, key=pair_counts.get)` to find
              the most frequent pair.
        """
        raise NotImplementedError("TODO: implement train")

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.

        Steps:
            1. Convert text to a list of single-byte tokens
               (same encoding as training).
            2. Apply each merge in `self.merges` in order, replacing
               every occurrence of that pair.
            3. Convert the final list of bytes tokens to IDs using
               `self.inverse_vocab`.

        Args:
            text: The string to encode.

        Returns:
            A list of integer token IDs.

        Hints:
            - The merge logic here is very similar to `_merge_pair`,
              but applied to a flat list of tokens instead of a dict.
            - Apply merges in the order they were learned (same order
              as `self.merges`).
        """
        raise NotImplementedError("TODO: implement encode")

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.

        Hints:
            - Look up each ID in `self.vocab` to get its bytes.
            - Concatenate all bytes with `b"".join(...)`.
            - Decode the result with `.decode("utf-8", errors="replace")`.
        """
        raise NotImplementedError("TODO: implement decode")
