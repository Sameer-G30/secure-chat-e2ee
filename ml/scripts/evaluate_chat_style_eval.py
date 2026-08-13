"""Score the trained baseline out-of-domain, on the hand-curated chat-style set.

This fits the same baseline pipeline on the full email/SMS training corpus
(never on the chat-style set: see data/label-schema.yaml's
evaluation_policy) and reports precision/recall/F1/confusion matrix on the
chat-style set only, so the README can state honestly how the register
shift affects real performance.

Usage (from ml/):
    uv run python scripts/evaluate_chat_style_eval.py
"""

# Import json to persist the out-of-domain metrics report as a checked-in artifact.
import json

# Import Path for portable input/output locations.
from pathlib import Path

from secure_chat_ml.baseline import (
    build_pipeline,
    evaluate_external,
    load_chat_style_eval_set,
    load_processed_corpora,
)

_DEFAULT_PROCESSED_DIR = Path("data/processed")
_DEFAULT_CHAT_EVAL_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")
_DEFAULT_REPORTS_DIR = Path("reports")


def main() -> None:
    """Train once on in-domain data, then score against the chat-style eval set."""

    # Train on every row of the combined in-domain corpus: this run's whole purpose
    # is measuring out-of-domain generalization, so there is no in-domain test split to protect.
    combined = load_processed_corpora(_DEFAULT_PROCESSED_DIR)
    pipeline = build_pipeline()
    pipeline.fit(combined["text"], combined["label"])

    # Load and score the hand-curated set; evaluate_external never calls .fit(...) on it.
    chat_eval_df = load_chat_style_eval_set(_DEFAULT_CHAT_EVAL_PATH)
    result = evaluate_external(pipeline, chat_eval_df)

    _DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = _DEFAULT_REPORTS_DIR / "chat_style_eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "trained_on_rows": len(combined),
                "chat_eval_rows": result.test_rows,
                "classification_report": result.classification_report,
                "confusion_matrix": result.confusion_matrix,
                "confusion_matrix_labels": ["legitimate", "scam"],
            },
            indent=2,
        )
    )

    scam_metrics = result.classification_report["scam"]
    legit_metrics = result.classification_report["legitimate"]
    print(f"Trained on {len(combined)} in-domain rows; evaluated on {result.test_rows} chat rows")
    print(
        "Scam class    -> precision: "
        f"{scam_metrics['precision']:.3f}  recall: {scam_metrics['recall']:.3f}  "
        f"f1: {scam_metrics['f1-score']:.3f}"
    )
    print(
        "Legitimate    -> precision: "
        f"{legit_metrics['precision']:.3f}  recall: {legit_metrics['recall']:.3f}  "
        f"f1: {legit_metrics['f1-score']:.3f}"
    )
    print(f"Confusion matrix ([[TN, FP], [FN, TP]]): {result.confusion_matrix}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
