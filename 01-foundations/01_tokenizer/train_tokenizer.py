"""Train a BPE tokenizer on a sample corpus and demonstrate its usage."""

from tokenizer import BPETokenizer


def main():
    corpus = """
    The quick brown fox jumps over the lazy dog.
    A language model learns to predict the next token.
    Byte Pair Encoding is a subword tokenization algorithm.
    """ * 50

    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.merges)}")

    text = "The quick brown fox"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"\nOriginal: {text}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {decoded}")
    print(f"Roundtrip OK: {text == decoded}")

    print(f"\nFirst 10 merges:")
    for i, (a, b) in enumerate(tokenizer.merges[:10]):
        print(f"  {i+1}. {a!r} + {b!r} -> {a+b!r}")


if __name__ == "__main__":
    main()
