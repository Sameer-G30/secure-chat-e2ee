"""Verify the chat-style evaluation-only dataset loads and scores without training on it."""

# Import Path for the temporary synthetic-corpus fixture directories.
from pathlib import Path

# Import pandas to build fixture DataFrames.
import pandas as pd

# Import pytest for fixture/raises support.
import pytest

from secure_chat_ml.baseline import (
    CLASS_NAMES,
    build_pipeline,
    evaluate_external,
    load_chat_style_eval_set,
)

_TRAIN_TEXTS_LEGIT = [
    "let's grab lunch tomorrow at noon",
    "can you send me the meeting notes",
    "happy birthday, hope you have a great day",
    "the report is attached, let me know if edits are needed",
] * 10
_TRAIN_TEXTS_SCAM = [
    "urgent: your account will be suspended, verify your password now",
    "you have won a prize, click this link to claim your reward",
    "congratulations, you are eligible for a free gift card, act now",
    "your bank account has been locked, confirm your login immediately",
] * 10


# Build a tiny fitted pipeline for tests that need one, without touching real corpora.
@pytest.fixture
def fitted_pipeline():
    """Return a pipeline fitted on tiny synthetic in-domain data."""

    frame = pd.DataFrame(
        {
            "text": _TRAIN_TEXTS_LEGIT + _TRAIN_TEXTS_SCAM,
            "label": [0] * len(_TRAIN_TEXTS_LEGIT) + [1] * len(_TRAIN_TEXTS_SCAM),
        }
    )
    pipeline = build_pipeline(max_features=1000)
    pipeline.fit(frame["text"], frame["label"])
    return pipeline


# Write a tiny synthetic chat-eval CSV shaped like the real hand-curated file.
@pytest.fixture
def synthetic_chat_eval_path(tmp_path: Path) -> Path:
    """Return a path to a small chat-eval CSV matching the real schema's columns."""

    frame = pd.DataFrame(
        {
            "message_id": ["chat-eval-legit-000", "chat-eval-scam-000"],
            "text": [
                "hey are we still on for lunch tomorrow?",
                "URGENT: your account will be suspended, verify your password now",
            ],
            "label": [0, 1],
            "original_label": ["legitimate_chat", "scam_chat"],
            "source": ["chat_style_eval_v1", "chat_style_eval_v1"],
            "split": ["eval_only", "eval_only"],
        }
    )
    path = tmp_path / "chat_style_eval_v1.csv"
    frame.to_csv(path, index=False)
    return path


# Confirm loading a missing eval set fails loudly, telling the user how to build it.
def test_load_chat_style_eval_set_requires_the_file_to_exist(tmp_path: Path) -> None:
    """Assert a missing chat-eval CSV raises FileNotFoundError with a helpful message."""

    missing_path = tmp_path / "does-not-exist.csv"
    with pytest.raises(FileNotFoundError, match="build_chat_style_eval_set.py"):
        load_chat_style_eval_set(missing_path)


# Confirm loading trims whitespace the same way the training loader does.
def test_load_chat_style_eval_set_strips_text(
    tmp_path: Path, synthetic_chat_eval_path: Path
) -> None:
    """Assert loaded text has no leading/trailing whitespace."""

    frame = load_chat_style_eval_set(synthetic_chat_eval_path)
    assert all(text == text.strip() for text in frame["text"])
    assert len(frame) == 2


# Confirm evaluate_external only predicts, reporting the metrics the spec requires.
def test_evaluate_external_reports_metrics_without_fitting(
    fitted_pipeline, synthetic_chat_eval_path: Path
) -> None:
    """Assert evaluate_external scores an out-of-domain set using only .predict()."""

    chat_eval_df = load_chat_style_eval_set(synthetic_chat_eval_path)
    result = evaluate_external(fitted_pipeline, chat_eval_df)

    # train_rows is always 0 here: this function must never fit anything.
    assert result.train_rows == 0
    assert result.test_rows == len(chat_eval_df)
    for class_name in CLASS_NAMES:
        class_metrics = result.classification_report[class_name]
        assert 0.0 <= class_metrics["precision"] <= 1.0
        assert 0.0 <= class_metrics["recall"] <= 1.0
    assert len(result.confusion_matrix) == 2
    assert sum(sum(row) for row in result.confusion_matrix) == len(chat_eval_df)
