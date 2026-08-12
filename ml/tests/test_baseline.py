"""Exercise the baseline pipeline against tiny synthetic data, not real corpora.

These tests never require the multi-gigabyte downloaded datasets, so they run
safely in CI without network access or the ml/data/raw fixtures.
"""

# Import Path for the temporary synthetic-corpus fixture directory.
from pathlib import Path

# Import pandas to build the synthetic corpus fixture DataFrame.
import pandas as pd

# Import pytest's tmp_path fixture typing implicitly through the function signature.
import pytest

from secure_chat_ml.baseline import (
    CLASS_NAMES,
    build_pipeline,
    evaluate,
    load_processed_corpora,
    save_confusion_matrix_plot,
    stratified_split,
)

# Build enough repeated-pattern rows that TF-IDF has real signal to separate classes.
_LEGITIMATE_TEXTS = [
    "let's grab lunch tomorrow at noon",
    "can you send me the meeting notes",
    "happy birthday, hope you have a great day",
    "the report is attached, let me know if edits are needed",
    "see you at the gym later this evening",
    "thanks for helping me move last weekend",
    "the weather looks nice for our hike on saturday",
    "here is the recipe you asked about",
    "let's reschedule our call to friday afternoon",
    "great job on the presentation today",
] * 6
_SCAM_TEXTS = [
    "urgent: your account will be suspended, verify your password now",
    "you have won a prize, click this link to claim your reward",
    "congratulations, you are eligible for a free gift card, act now",
    "your bank account has been locked, confirm your login immediately",
    "limited time offer, wire money now to secure this deal",
    "verify your identity within 24 hours or lose access to your account",
    "you have an unclaimed refund, provide your card details to receive it",
    "final notice: update your payment information to avoid suspension",
    "click here immediately to unlock your frozen account",
    "act now, this exclusive investment opportunity expires today",
] * 6


# Write a tiny synthetic corpus CSV shaped like the real label-schema output.
@pytest.fixture
def synthetic_processed_dir(tmp_path: Path) -> Path:
    """Return a directory containing one small, schema-shaped corpus CSV."""

    rows = []
    for index, text in enumerate(_LEGITIMATE_TEXTS):
        rows.append(
            {
                "message_id": f"synthetic-legit-{index:04d}",
                "text": text,
                "label": 0,
                "original_label": "ham",
                "source": "synthetic_test_corpus",
                "split": "unassigned",
            }
        )
    for index, text in enumerate(_SCAM_TEXTS):
        rows.append(
            {
                "message_id": f"synthetic-scam-{index:04d}",
                "text": text,
                "label": 1,
                "original_label": "scam",
                "source": "synthetic_test_corpus",
                "split": "unassigned",
            }
        )
    frame = pd.DataFrame(rows)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    frame.to_csv(processed_dir / "synthetic.csv", index=False)
    return processed_dir


# Confirm loading raises a clear error rather than silently training on nothing.
def test_load_processed_corpora_requires_at_least_one_csv(tmp_path: Path) -> None:
    """Assert an empty processed directory fails loudly."""

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_processed_corpora(empty_dir)


# Confirm loading drops empty text and exact duplicates.
def test_load_processed_corpora_drops_empty_and_duplicate_rows(tmp_path: Path) -> None:
    """Assert dirty synthetic rows are cleaned before training ever sees them."""

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    frame = pd.DataFrame(
        {
            "message_id": ["a", "b", "c", "d"],
            "text": ["hello there", "hello there", "   ", "legit unique text"],
            "label": [0, 0, 1, 0],
            "original_label": ["ham", "ham", "spam", "ham"],
            "source": ["s", "s", "s", "s"],
            "split": ["unassigned"] * 4,
        }
    )
    frame.to_csv(processed_dir / "dirty.csv", index=False)

    combined = load_processed_corpora(processed_dir)

    # The duplicate "hello there" row and the whitespace-only row must both be gone.
    assert len(combined) == 2
    assert set(combined["text"]) == {"hello there", "legit unique text"}


# Confirm the split preserves class proportions in both partitions.
def test_stratified_split_preserves_class_balance(synthetic_processed_dir: Path) -> None:
    """Assert both splits contain both classes at roughly the source proportion."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, test_df = stratified_split(combined, test_size=0.2, random_state=42)

    # Both splits must contain both classes; stratification exists precisely for this.
    assert set(train_df["label"].unique()) == {0, 1}
    assert set(test_df["label"].unique()) == {0, 1}
    # The test split should be close to the requested 20% of the combined rows.
    assert abs(len(test_df) / len(combined) - 0.2) < 0.05


# Confirm the full train -> evaluate path reports the metrics the spec requires.
def test_evaluate_reports_precision_recall_f1_and_confusion_matrix(
    synthetic_processed_dir: Path,
) -> None:
    """Assert evaluate() returns per-class metrics and a well-formed confusion matrix."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, test_df = stratified_split(combined, test_size=0.3, random_state=42)
    pipeline = build_pipeline(max_features=1000)

    result = evaluate(pipeline, train_df, test_df)

    # Every class name from the shared schema must have a precision/recall/F1 entry.
    for class_name in CLASS_NAMES:
        class_metrics = result.classification_report[class_name]
        assert 0.0 <= class_metrics["precision"] <= 1.0
        assert 0.0 <= class_metrics["recall"] <= 1.0
        assert 0.0 <= class_metrics["f1-score"] <= 1.0
    # The confusion matrix must be a 2x2 grid whose total equals the test set size.
    assert len(result.confusion_matrix) == 2
    assert all(len(row) == 2 for row in result.confusion_matrix)
    assert sum(sum(row) for row in result.confusion_matrix) == result.test_rows
    # This synthetic corpus is easy to separate; the baseline should do far better than chance.
    assert result.classification_report["scam"]["f1-score"] > 0.7


# Confirm the confusion-matrix plot writes a real, non-empty image file.
def test_save_confusion_matrix_plot_writes_a_file(
    tmp_path: Path, synthetic_processed_dir: Path
) -> None:
    """Assert the saved PNG exists and has non-trivial size."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, test_df = stratified_split(combined, test_size=0.3, random_state=42)
    pipeline = build_pipeline(max_features=1000)
    result = evaluate(pipeline, train_df, test_df)

    output_path = tmp_path / "reports" / "confusion_matrix.png"
    save_confusion_matrix_plot(result, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
