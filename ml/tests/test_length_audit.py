"""Exercise length-audit helpers against synthetic lengths, never the 71k corpus."""

# Import numpy to build tiny length vectors.
import numpy as np

# Import pandas for the character-length filter test.
import pandas as pd

from secure_chat_ml.length_audit import (
    character_lengths,
    chunking_rejection_reason,
    filter_by_character_length,
    summarize_lengths,
)


def test_summarize_lengths_counts_overflow_and_percentiles() -> None:
    lengths = np.asarray([10, 20, 30, 40, 300], dtype=np.int64)
    summary = summarize_lengths(lengths, caps=(128, 256))
    assert summary["n"] == 5
    assert summary["overflow_by_cap"]["128"] == 1
    assert summary["overflow_by_cap"]["256"] == 1
    assert summary["percentiles"]["100"] == 300.0


def test_character_lengths_match_python_len() -> None:
    texts = ["hi", "hello world"]
    assert character_lengths(texts).tolist() == [2, 11]


def test_filter_by_character_length_keeps_dm_sized_rows() -> None:
    frame = pd.DataFrame(
        {
            "text": ["short", "a" * 50, "b" * 400],
            "label": [0, 0, 1],
        }
    )
    filtered = filter_by_character_length(frame, max_chars=100)
    assert list(filtered["text"]) == ["short", "a" * 50]


def test_chunking_rejection_reason_mentions_false_alarms_not_truncation() -> None:
    text = chunking_rejection_reason(297, 49958, 40.0)
    assert "rejected" in text.lower()
    assert "256" in text
    assert "false-alarm" in text
