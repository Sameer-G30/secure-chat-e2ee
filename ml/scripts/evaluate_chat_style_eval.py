"""Score the trained baseline out-of-domain on the locked chat-style eval set.

Fits the same TF-IDF + URL + Logistic Regression pipeline on TRAIN+VAL of
the LLM chat-register training data (never on data/chat_eval/chat_style_eval_v1.csv;
see data/label-schema.yaml evaluation_policy.chat_style_eval_training_allowed:
false). Applies the FROZEN C and decision threshold from
reports/baseline_metrics.json; does not retune on the locked 200 rows.

Usage (from ml/):
    uv run python scripts/evaluate_chat_style_eval.py
"""

# Import argparse so a reviewer can point at fixture directories in tests.
import argparse

# Import json to read the frozen threshold and persist the OOD metrics report.
import json

# Import Path for portable input/output locations.
from pathlib import Path

# Import pandas to concatenate TRAIN+VAL without touching TEST or chat eval.
import pandas as pd

from secure_chat_ml.baseline import (
    build_pipeline,
    evaluate_external,
    hyperparameters_from_mapping,
    infer_rewrite_method,
    load_chat_style_eval_set,
    load_processed_corpora,
    stratified_split,
)

# Default to LLM chat-register CSVs, matching train_baseline.py.
_DEFAULT_PROCESSED_DIR = Path("data/processed_chat_llm")
# The locked evaluation-only file; never passed to Pipeline.fit.
_DEFAULT_CHAT_EVAL_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")
# Read frozen C/threshold from the TEST report written by train_baseline.py.
_DEFAULT_REPORTS_DIR = Path("reports")


# Parse command-line arguments controlling the out-of-domain scoring run.
def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments with repository-relative defaults."""

    # Build a parser from this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__)
    # Allow overriding the training-text directory (must not be data/chat_eval).
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Training-text directory (default: processed_dir from baseline_metrics.json).",
    )
    # Allow pointing at a synthetic chat-eval CSV in tests.
    parser.add_argument(
        "--chat-eval-path",
        type=Path,
        default=_DEFAULT_CHAT_EVAL_PATH,
        help="Locked chat-style eval CSV (default: data/chat_eval/chat_style_eval_v1.csv).",
    )
    # Reports directory containing the frozen baseline_metrics.json.
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_DEFAULT_REPORTS_DIR,
        help="Directory with baseline_metrics.json and for the OOD report (default: reports).",
    )
    # Allow writing v1 and v2 reports side by side without overwriting.
    parser.add_argument(
        "--metrics-filename",
        type=str,
        default="chat_style_eval_metrics.json",
        help="JSON filename under --reports-dir (default: chat_style_eval_metrics.json).",
    )
    # Return the populated namespace for main().
    return parser.parse_args()


# Load the frozen operating point written after validation-only tuning.
def load_frozen_choices(reports_dir: Path) -> dict:
    """Return the baseline_metrics.json payload, failing if training has not been run."""

    # Resolve the TEST report that also stores chosen_C and chosen_threshold.
    metrics_path = reports_dir / "baseline_metrics.json"
    # Fail loudly rather than silently retuning or defaulting to 0.5.
    if not metrics_path.exists():
        # Tell the operator to freeze validation choices first.
        raise FileNotFoundError(
            f"No frozen baseline report at {metrics_path}. "
            "Run scripts/train_baseline.py first; this script must not retune "
            "on chat_style_eval_v1.csv."
        )
    # Parse the JSON report.
    payload = json.loads(metrics_path.read_text())
    # Require the frozen threshold key so an old 80/20 report cannot be reused silently.
    if "chosen_threshold" not in payload or "chosen_C" not in payload:
        # Ask for a regenerated report from the new training protocol.
        raise ValueError(
            f"{metrics_path} is missing chosen_threshold/chosen_C. "
            "Re-run scripts/train_baseline.py with the 70/20/10 protocol."
        )
    # Return the full payload so split seeds can be reconstructed.
    return payload


# Fit on TRAIN+VAL of the training-text directory, then score the locked eval set only.
def main() -> None:
    """Train on in-domain TRAIN+VAL, then score the locked chat-style eval set."""

    # Parse CLI arguments.
    args = parse_args()
    # Load the frozen C and threshold chosen on validation (never on the 200 rows).
    frozen = load_frozen_choices(args.reports_dir)
    # Prefer the processed_dir recorded at train time unless the CLI overrode it.
    processed_dir = args.processed_dir or Path(frozen.get("processed_dir", _DEFAULT_PROCESSED_DIR))
    # Reconstruct the same nested split so TRAIN+VAL matches the training run.
    random_state = int(frozen.get("random_state", 42))
    # Read the split fractions recorded at train time (defaults are 70/20/10).
    train_size = float(frozen.get("train_size", 0.7))
    # Validation fraction used only to reconstruct which rows were TRAIN+VAL.
    val_size = float(frozen.get("val_size", 0.2))
    # Test fraction is reconstructed so those rows stay out of this fit too.
    test_size = float(frozen.get("test_size", 0.1))
    # Read the frozen logistic C.
    chosen_C = float(frozen["chosen_C"])
    # Read the frozen decision threshold; do not search it on chat eval.
    chosen_threshold = float(frozen["chosen_threshold"])
    # Rebuild the same TF-IDF/URL/logistic knobs used at train time (old reports default).
    hyperparameters = hyperparameters_from_mapping(frozen)
    # Read the TF-IDF vocabulary cap so this fit matches the reported model.
    max_features = int(hyperparameters.max_features)

    # Refuse to treat the locked eval directory as a training source.
    if "chat_eval" in processed_dir.parts:
        # Honor evaluation_policy.chat_style_eval_training_allowed: false.
        raise ValueError(
            f"Refusing to fit on {processed_dir}: chat_style_eval_training_allowed is false."
        )

    # Load chat-register (or original) training text; this glob never sees chat_eval/.
    combined = load_processed_corpora(processed_dir)
    # Honor a length-mismatch experiment recorded in baseline_metrics.json.
    max_chars = frozen.get("max_chars")
    if max_chars is not None:
        from secure_chat_ml.length_audit import filter_by_character_length

        combined = filter_by_character_length(combined, max_chars=int(max_chars))
    # Reconstruct TRAIN/VAL/TEST with the same seed used by train_baseline.py.
    train_df, val_df, test_df = stratified_split(
        combined,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    # Fit on TRAIN+VAL only; TEST stays held out and chat eval is never concatenated.
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    # Build the pipeline at the frozen C, including the local URL-feature branch.
    pipeline = build_pipeline(
        max_features=max_features,
        C=chosen_C,
        hyperparameters=hyperparameters,
    )
    # Fit on in-domain TRAIN+VAL text/labels only.
    pipeline.fit(fit_df["text"], fit_df["label"])

    # Load the locked eval set; evaluate_external never calls .fit(...) on it.
    chat_eval_df = load_chat_style_eval_set(args.chat_eval_path)
    # Score with the frozen threshold; do not retune on these rows.
    result = evaluate_external(pipeline, chat_eval_df, threshold=chosen_threshold)

    # Ensure the reports directory exists before writing the OOD artifact.
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    # Persist the out-of-domain metrics next to the in-domain TEST report.
    metrics_path = args.reports_dir / args.metrics_filename
    # Record that the 200-row file was never used for fitting or threshold search.
    metrics_path.write_text(
        json.dumps(
            {
                "trained_on_rows": len(fit_df),
                "trained_on": f"train_plus_val_{processed_dir.name}",
                "processed_dir": str(processed_dir),
                "rewrite_method": frozen.get(
                    "rewrite_method", infer_rewrite_method(processed_dir)
                ),
                "held_out_in_domain_test_rows": len(test_df),
                "chat_eval_rows": result.test_rows,
                "chosen_C": chosen_C,
                "chosen_threshold": chosen_threshold,
                "threshold_source": "reports/baseline_metrics.json (validation-frozen)",
                "retuned_on_chat_eval": False,
                "chat_style_eval_training_allowed": False,
                "chat_eval_path": str(args.chat_eval_path),
                "classification_report": result.classification_report,
                "confusion_matrix": result.confusion_matrix,
                "confusion_matrix_labels": ["legitimate", "scam"],
                "live_url_reputation": False,
            },
            indent=2,
        )
    )

    # Print a concise human-readable summary to the terminal.
    scam_metrics = result.classification_report["scam"]
    # Also print legitimate metrics so false-positive cost is visible.
    legit_metrics = result.classification_report["legitimate"]
    # State clearly that chat eval was scored, not trained on.
    print(
        f"Fitted on {len(fit_df)} TRAIN+VAL rows; "
        f"evaluated on {result.test_rows} locked chat-eval rows "
        f"(threshold={chosen_threshold}, C={chosen_C})"
    )
    # Show scam-class OOD metrics for the README comparison vs the old 0.800 recall.
    print(
        "Scam class    -> precision: "
        f"{scam_metrics['precision']:.3f}  recall: {scam_metrics['recall']:.3f}  "
        f"f1: {scam_metrics['f1-score']:.3f}"
    )
    # Show legitimate-class OOD metrics.
    print(
        "Legitimate    -> precision: "
        f"{legit_metrics['precision']:.3f}  recall: {legit_metrics['recall']:.3f}  "
        f"f1: {legit_metrics['f1-score']:.3f}"
    )
    # Show the OOD confusion matrix in [[TN, FP], [FN, TP]] order.
    print(f"Confusion matrix ([[TN, FP], [FN, TP]]): {result.confusion_matrix}")
    # Point the operator at the written artifact.
    print(f"Wrote {metrics_path}")


# Run the OOD evaluation only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
