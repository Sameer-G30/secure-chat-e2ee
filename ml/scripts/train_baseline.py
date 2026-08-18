"""CLI entry point: train the TF-IDF + URL baseline and write metrics reports.

Fits on TRAIN only, tunes C and the decision threshold on VALIDATION only,
then scores TEST once. Default training text is the chat-register rewrite
in data/processed_chat; pass --processed-dir data/processed to compare
against original email/SMS.

Usage (from ml/):
    uv run python scripts/train_baseline.py
"""

# Import argparse to expose the tunable knobs a reviewer might want to vary.
import argparse

# Import json to persist the metrics reports as checked-in artifacts.
import json

# Import Path for portable input/output locations.
from pathlib import Path

# Import the testable training/evaluation logic this script only orchestrates.
from secure_chat_ml.baseline import (
    DEFAULT_C_GRID,
    DEFAULT_LEGIT_PRECISION_FLOOR,
    DEFAULT_THRESHOLD_GRID,
    ThresholdTuningResult,
    build_pipeline,
    evaluate,
    evaluate_external,
    load_processed_corpora,
    save_confusion_matrix_plot,
    stratified_split,
    tune_on_validation,
)

# Default to chat-register training text after scripts/rewrite_chat_register.py has been run.
_DEFAULT_PROCESSED_DIR = Path("data/processed_chat")
# Default to the repository-relative reports directory every other ml/ script already uses.
_DEFAULT_REPORTS_DIR = Path("reports")


# Convert a numpy-ish value inside a classification_report into JSON-safe types.
def _jsonify(value: object) -> object:
    """Return a JSON-serializable copy of nested sklearn report structures."""

    # Recurse into dictionaries produced by classification_report(..., output_dict=True).
    if isinstance(value, dict):
        # Convert every nested value the same way.
        return {str(key): _jsonify(item) for key, item in value.items()}
    # Recurse into confusion-matrix rows.
    if isinstance(value, list):
        # Convert every cell.
        return [_jsonify(item) for item in value]
    # Cast numpy scalars (and numpy bools) to plain Python types.
    if hasattr(value, "item") and callable(value.item):
        # np.float64(0.9).item() → Python float.
        try:
            # Prefer the scalar's native Python type.
            return value.item()
        except (ValueError, AttributeError):
            # Fall through to float/int coercion when .item() is not a scalar.
            pass
    # Leave strings and bools unchanged.
    if isinstance(value, (str, bool)) or value is None:
        # JSON already supports these types.
        return value
    # Cast remaining numbers to Python int or float.
    if isinstance(value, int) and not isinstance(value, bool):
        # Keep integers as integers (support counts).
        return int(value)
    # Treat everything else numeric as float (precision/recall/F1).
    if isinstance(value, float):
        # Return a plain Python float.
        return float(value)
    # Last resort: stringify unknown objects rather than crashing JSON dumps.
    return str(value)


# Parse command-line arguments controlling the training run.
def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments with the same defaults used by local runs."""

    # Build a parser from this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to rewritten chat-register CSVs; original email/SMS remain available.
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_DEFAULT_PROCESSED_DIR,
        help=(
            "Directory containing normalized per-corpus CSVs "
            "(default: data/processed_chat). Pass data/processed to train on "
            "original email/SMS for comparison."
        ),
    )
    # Write TEST metrics, VAL metrics, and the confusion-matrix figure here.
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_DEFAULT_REPORTS_DIR,
        help="Directory to write metrics JSON and confusion_matrix.png (default: reports).",
    )
    # Stratified train fraction of the combined corpus.
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Fraction of rows used for fitting (default: 0.7).",
    )
    # Stratified validation fraction used only for C/threshold selection.
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for threshold/C tuning (default: 0.2).",
    )
    # Stratified test fraction scored once after freezing validation choices.
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of rows held out for the final test report (default: 0.1).",
    )
    # Seed controlling the nested stratified split for reproducibility.
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed controlling the train/val/test split (default: 42).",
    )
    # Maximum TF-IDF vocabulary size.
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum TF-IDF vocabulary size (default: 50000).",
    )
    # Enable or disable the validation-only C/threshold search (default on).
    parser.add_argument(
        "--tune-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search C and the decision threshold on validation (default: on).",
    )
    # Return the populated namespace for main().
    return parser.parse_args()


# Serialize the frozen validation choices plus the VAL classification report.
def _val_metrics_payload(tuning: ThresholdTuningResult, args: argparse.Namespace) -> dict:
    """Return the JSON body written to reports/val_metrics.json."""

    # Include the chosen operating point, why it was chosen, and VAL metrics.
    return {
        "selection_rule": tuning.selection_rule,
        "selection_reason": tuning.selection_reason,
        "legit_precision_floor": tuning.legit_precision_floor,
        "floor_feasible": tuning.floor_feasible,
        "chosen_C": tuning.C,
        "chosen_threshold": tuning.threshold,
        "grid_C": tuning.grid_C,
        "grid_thresholds": tuning.grid_thresholds,
        "val_rows": tuning.val_rows,
        "classification_report": _jsonify(tuning.classification_report),
        "confusion_matrix": tuning.confusion_matrix,
        "confusion_matrix_labels": ["legitimate", "scam"],
        "random_state": args.random_state,
        "tune_threshold": args.tune_threshold,
        "live_url_reputation": False,
        "url_features": True,
        "chat_style_eval_used_for_tuning": False,
    }


# Run the full load → split → tune on val → score test → report pipeline.
def main() -> None:
    """Train the baseline once and write TEST metrics plus a confusion-matrix image."""

    # Parse CLI arguments (defaults match the documented from-ml/ command).
    args = parse_args()

    # Load every rewritten (or original) corpus into one combined dataset.
    combined = load_processed_corpora(args.processed_dir)
    # Split with class balance preserved in train, validation, and test.
    train_df, val_df, test_df = stratified_split(
        combined,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    # Tune C and the decision threshold on VALIDATION only, unless the operator opted out.
    if args.tune_threshold:
        # Search the documented grids; never look at TEST or the locked chat eval set.
        tuning = tune_on_validation(
            train_df,
            val_df,
            max_features=args.max_features,
            C_grid=DEFAULT_C_GRID,
            threshold_grid=DEFAULT_THRESHOLD_GRID,
            legit_precision_floor=DEFAULT_LEGIT_PRECISION_FLOOR,
            random_state=args.random_state,
        )
    else:
        # Keep C=1.0 and threshold=0.5, but still score VAL at that default for the audit file.
        default_pipeline = build_pipeline(max_features=args.max_features, C=1.0)
        # Fit on TRAIN only even when tuning is disabled.
        default_pipeline.fit(train_df["text"], train_df["label"])
        # Score VAL at the default 0.5 threshold so val_metrics.json still exists.
        val_eval = evaluate_external(default_pipeline, val_df, threshold=0.5)
        # Record that the operator skipped the search.
        tuning = ThresholdTuningResult(
            C=1.0,
            threshold=0.5,
            selection_reason="default_no_tune",
            selection_rule="operator disabled tuning; C=1.0 and threshold=0.5",
            legit_precision_floor=DEFAULT_LEGIT_PRECISION_FLOOR,
            floor_feasible=False,
            classification_report=val_eval.classification_report,
            confusion_matrix=val_eval.confusion_matrix,
            val_rows=len(val_df),
            grid_C=[1.0],
            grid_thresholds=[0.5],
        )
    # Build a fresh pipeline at the frozen C so TEST scoring does not reuse VAL-fitted heads.
    pipeline = build_pipeline(max_features=args.max_features, C=tuning.C)
    # Fit on TRAIN and score TEST once with the frozen threshold.
    result = evaluate(pipeline, train_df, test_df, threshold=tuning.threshold)

    # Ensure the reports directory exists before writing any artifact.
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    # Persist TEST metrics as the numbers the README must quote.
    metrics_path = args.reports_dir / "baseline_metrics.json"
    # Include the frozen choices so evaluate_chat_style_eval.py can reuse them.
    metrics_path.write_text(
        json.dumps(
            {
                "train_rows": result.train_rows,
                "val_rows": len(val_df),
                "test_rows": result.test_rows,
                "chosen_C": tuning.C,
                "chosen_threshold": tuning.threshold,
                "selection_reason": tuning.selection_reason,
                "selection_rule": tuning.selection_rule,
                "legit_precision_floor": tuning.legit_precision_floor,
                "floor_feasible": tuning.floor_feasible,
                "classification_report": _jsonify(result.classification_report),
                "confusion_matrix": result.confusion_matrix,
                "confusion_matrix_labels": ["legitimate", "scam"],
                "source_counts_in_test_split": result.source_counts,
                "random_state": args.random_state,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "max_features": args.max_features,
                "processed_dir": str(args.processed_dir),
                "rewrite_method": "rule_based_v1"
                if "processed_chat" in Path(args.processed_dir).parts
                else "none",
                "url_features": True,
                "live_url_reputation": False,
                "chat_style_eval_used_for_training": False,
                "chat_style_eval_used_for_tuning": False,
            },
            indent=2,
        )
    )
    # Persist VAL metrics separately so the threshold search remains auditable.
    val_path = args.reports_dir / "val_metrics.json"
    # Write the validation payload next to the TEST report.
    val_path.write_text(json.dumps(_val_metrics_payload(tuning, args), indent=2))
    # Save the TEST confusion-matrix heatmap next to the JSON reports.
    save_confusion_matrix_plot(result, args.reports_dir / "confusion_matrix.png")

    # Print a concise human-readable summary to the terminal.
    scam_metrics = result.classification_report["scam"]
    # Show split sizes so a reviewer can confirm 70/20/10 at a glance.
    print(
        f"Train rows: {result.train_rows}  Val rows: {len(val_df)}  "
        f"Test rows: {result.test_rows}"
    )
    # Show the frozen operating point chosen on validation.
    print(
        f"Frozen C: {tuning.C}  Frozen threshold: {tuning.threshold}  "
        f"Reason: {tuning.selection_reason}"
    )
    # Show TEST scam metrics — these are the numbers the README must quote.
    print(
        "TEST scam class -> precision: "
        f"{scam_metrics['precision']:.3f}  recall: {scam_metrics['recall']:.3f}  "
        f"f1: {scam_metrics['f1-score']:.3f}"
    )
    # Show the TEST confusion matrix in [[TN, FP], [FN, TP]] order.
    print(f"TEST confusion matrix ([[TN, FP], [FN, TP]]): {result.confusion_matrix}")
    # Point the operator at the written artifacts.
    print(
        f"Wrote {metrics_path}, {val_path}, and {args.reports_dir / 'confusion_matrix.png'}"
    )


# Run the training pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
