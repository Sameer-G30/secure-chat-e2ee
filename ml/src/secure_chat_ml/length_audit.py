"""Length-audit helpers for DistilBERT WordPiece and character-length stats.

The published DistilBERT recipe tokenizes with truncation=True, max_length=256,
padding=False (see secure_chat_ml.distilbert.tokenize_texts). This module does
not train; it only measures how often that cap actually fires, so the
pre-deployment review can reject chunked parallel inference with numbers.
"""

# Import statistics helpers for percentile summaries without pulling sklearn.
from typing import Any

# Import numpy to compute percentiles on token/character length vectors.
import numpy as np

# Import pandas so callers can pass a combined corpus DataFrame.
import pandas as pd

# Percentiles reported in ml/reports/length_audit.json.
_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0, 100.0)

# Caps DistilBERT was swept at; 256 is the published serving length.
DISTILBERT_LENGTH_CAPS: tuple[int, ...] = (128, 256, 384, 512)


# Summarize a 1-D length vector as JSON-serializable percentiles plus overflow counts.
def summarize_lengths(
    lengths: np.ndarray,
    caps: tuple[int, ...] = DISTILBERT_LENGTH_CAPS,
) -> dict[str, Any]:
    """Return n, mean, percentiles, and how many values exceed each cap."""

    # An empty split still produces a report so a missing file is obvious.
    if lengths.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "percentiles": {str(int(p) if p.is_integer() else p): 0.0 for p in _PERCENTILES},
            "overflow_by_cap": {str(cap): 0 for cap in caps},
        }
    # Cast to float so percentile math is stable for integer token counts.
    values = lengths.astype(np.float64)
    # Build the percentile map quoted by length_audit.md.
    percentiles = {
        str(int(p) if float(p).is_integer() else p): float(np.percentile(values, p))
        for p in _PERCENTILES
    }
    # Count rows that would truncate at each DistilBERT max_length.
    overflow = {str(cap): int(np.sum(values > cap)) for cap in caps}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "percentiles": percentiles,
        "overflow_by_cap": overflow,
    }


# Count Unicode characters per message (TF-IDF has no token cap; this is the DM-length view).
def character_lengths(texts: list[str]) -> np.ndarray:
    """Return one integer character count per text, as a numpy vector."""

    return np.asarray([len(text) for text in texts], dtype=np.int64)


# Filter a labeled corpus to DM-like character lengths for the length-mismatch experiment.
def filter_by_character_length(
    frame: pd.DataFrame,
    *,
    max_chars: int,
    min_chars: int = 1,
) -> pd.DataFrame:
    """Return rows whose text length is in [min_chars, max_chars], inclusive.

    Used to retrain TF-IDF on a DM-length-filtered corpus without touching the
    locked chat_style_eval_v1.csv file. Callers must still split TRAIN/VAL/TEST
    on the filtered frame; this function never looks at labels for filtering.
    """

    # Work on a copy so the caller's combined frame stays intact.
    filtered = frame.copy()
    # Measure Unicode length of the already-stripped text column.
    sizes = filtered["text"].astype(str).str.len()
    # Keep only rows inside the requested DM-length window.
    filtered = filtered[(sizes >= min_chars) & (sizes <= max_chars)]
    # Reset so downstream positional splits stay aligned.
    return filtered.reset_index(drop=True)


# Explain why chunked/sliding-window DistilBERT inference is rejected for this product.
def chunking_rejection_reason(
    train_overflow_at_256: int,
    train_n: int,
    chat_eval_p100_tokens: float,
) -> str:
    """Return the documented rejection of chunked parallel DistilBERT inference."""

    return (
        "Chunked (sliding-window) DistilBERT inference is rejected for this product. "
        f"Only {train_overflow_at_256}/{train_n} TRAIN rows overflow 256 WordPiece tokens "
        f"({(train_overflow_at_256 / train_n * 100) if train_n else 0:.2f}%), and the locked "
        f"chat-style eval set's p100 token length is {chat_eval_p100_tokens:.1f} — well under "
        "the serving cap. Splitting a DM into windows would invent a second, untrained "
        "aggregation rule (max/mean/any-window-warn) on a domain that does not need it, "
        "and would multiply the 345 ms DistilBERT cost in the browser. The false-alarm "
        "problem is not truncation; it is the 0.85 in-domain legitimate-recall floor."
    )
