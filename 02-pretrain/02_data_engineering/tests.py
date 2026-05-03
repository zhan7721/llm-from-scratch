import pytest
from data_engineering import MinHashDeduplicator, QualityFilter, DataMixer


def test_minhash_finds_duplicates():
    dedup = MinHashDeduplicator(num_hashes=64, threshold=0.7)
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog!",  # near-duplicate
        "A completely different document about something else entirely.",
    ]
    result = dedup.deduplicate(docs)
    assert len(result) == 2  # one duplicate removed


def test_minhash_keeps_unique():
    dedup = MinHashDeduplicator(num_hashes=64, threshold=0.8)
    docs = [
        "First document about science and physics.",
        "Second document about cooking and recipes.",
        "Third document about travel and adventure.",
    ]
    result = dedup.deduplicate(docs)
    assert len(result) == 3  # all unique


def test_minhash_empty():
    dedup = MinHashDeduplicator()
    assert dedup.deduplicate([]) == []


def test_quality_filter_removes_short():
    qf = QualityFilter(min_words=5)
    docs = ["Short.", "This is a longer document with enough words to pass the filter."]
    result = qf.filter(docs)
    assert len(result) == 1


def test_quality_filter_removes_digit_heavy():
    qf = QualityFilter(max_digit_ratio=0.3)
    docs = [
        "12345 67890 1234 5678 9012 3456 7890",
        "This is a normal document with mostly text and few digits 123.",
    ]
    result = qf.filter(docs)
    assert len(result) == 1


def test_quality_filter_passes_good_text():
    qf = QualityFilter()
    doc = "This is a well-written document with enough words and good quality text."
    result = qf.filter([doc])
    assert len(result) == 1


def test_data_mixer_respects_ratios():
    mixer = DataMixer({"a": 0.7, "b": 0.3})
    data = {"a": ["a"] * 100, "b": ["b"] * 100}
    result = mixer.mix(data)
    a_count = sum(1 for d in result if d == "a")
    b_count = sum(1 for d in result if d == "b")
    # Approximately 70/30 split
    assert a_count > b_count


def test_data_mixer_with_total_tokens():
    mixer = DataMixer({"a": 0.5, "b": 0.5})
    data = {"a": ["doc_a"] * 1000, "b": ["doc_b"] * 1000}
    result = mixer.mix(data, total_tokens=1000, tokens_per_doc=10)
    assert len(result) > 0
