"""CLI entry point: train the TF-IDF baseline and write a metrics report.

Usage (from ml/):
    uv run python scripts/train_baseline.py
"""

# Import argparse to expose the tunable knobs a reviewer might want to vary.
import argparse

# Import json to persist the metrics report as a checked-in artifact.
import json

# Import Path for portable input/output locations.
from pathlib import Path

# Import the testable training/evaluation logic this script only orchestrates.
from secure_chat_ml.baseline import (
    build_pipeline,
    evaluate,
    load_processed_corpora,
    save_confusion_matrix_plot,
    stratified_split,
)

# Default to the repository-relative locations every other ml/ script already uses.
_DEFAULT_PROCESSED_DIR = Path("data/processed")
_DEFAULT_REPORTS_DIR = Path("reports")


# Parse command-line arguments controlling the training run.
def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments with the same defaults used by CI-free local runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_DEFAULT_PROCESSED_DIR,
        help="Directory containing normalized per-corpus CSVs (default: data/processed).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_DEFAULT_REPORTS_DIR,
        help="Directory to write metrics.json and confusion_matrix.png (default: reports).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows held out for the stratified test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed controlling the train/test split for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum TF-IDF vocabulary size (default: 50000).",
    )
    return parser.parse_args()


# Run the full load -> split -> train -> evaluate -> report pipeline.
def main() -> None:
    """Train the baseline once and write metrics.json plus a confusion-matrix image."""

    args = parse_args()

    # Load every downloaded and normalized corpus into one combined dataset.
    combined = load_processed_corpora(args.processed_dir)
    # Split with the class balance preserved in both partitions.
    train_df, test_df = stratified_split(
        combined, test_size=args.test_size, random_state=args.random_state
    )
    # Build the untrained TF-IDF + Logistic Regression pipeline.
    pipeline = build_pipeline(max_features=args.max_features)
    # Fit on the training split and evaluate on the untouched test split.
    result = evaluate(pipeline, train_df, test_df)

    # Ensure the reports directory exists before writing any artifact.
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    # Persist the full metrics report as JSON for the README and future comparisons.
    metrics_path = args.reports_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "classification_report": result.classification_report,
                "confusion_matrix": result.confusion_matrix,
                "confusion_matrix_labels": ["legitimate", "scam"],
                "source_counts_in_test_split": result.source_counts,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "max_features": args.max_features,
            },
            indent=2,
        )
    )
    # Save the confusion-matrix heatmap next to the JSON report.
    save_confusion_matrix_plot(result, args.reports_dir / "confusion_matrix.png")

    # Print a concise human-readable summary to the terminal.
    scam_metrics = result.classification_report["scam"]
    print(f"Train rows: {result.train_rows}  Test rows: {result.test_rows}")
    print(
        "Scam class -> precision: "
        f"{scam_metrics['precision']:.3f}  recall: {scam_metrics['recall']:.3f}  "
        f"f1: {scam_metrics['f1-score']:.3f}"
    )
    print(f"Confusion matrix ([[TN, FP], [FN, TP]]): {result.confusion_matrix}")
    print(f"Wrote {metrics_path} and {args.reports_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
