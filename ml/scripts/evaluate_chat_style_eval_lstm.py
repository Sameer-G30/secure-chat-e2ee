"""Score a trained word BiLSTM checkpoint on the locked chat-style eval set.

Loads the frozen threshold from reports/lstm/test_metrics.json (chosen on
VALIDATION only) and the checkpoint under models/lstm/. Never fits or
retunes on data/chat_eval/chat_style_eval_v1.csv.

Usage (from ml/):
    uv run python scripts/evaluate_chat_style_eval_lstm.py
"""

# Import argparse so a reviewer can point at fixture directories in tests.
import argparse

# Import json to read the frozen threshold and persist the OOD metrics report.
import json

# Import Path for portable input/output locations.
from pathlib import Path

from secure_chat_ml.baseline import infer_rewrite_method, load_chat_style_eval_set
from secure_chat_ml.lstm import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_MAX_TOKENS,
    evaluate_from_proba,
    load_saved_classifier,
    predict_scam_proba,
    resolve_training_device,
)

# Default to LLM chat-register CSVs, matching train_lstm.py.
_DEFAULT_PROCESSED_NAME = "processed_chat_llm"

# The locked evaluation-only file; never passed to train_model().
_DEFAULT_CHAT_EVAL_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")

# Read frozen threshold from the TEST report written by train_lstm.py.
_DEFAULT_REPORTS_DIR = Path("reports/lstm")

# Default checkpoint location written by train_lstm.py.
_DEFAULT_MODEL_DIR = Path("models/lstm")


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


# Parse command-line arguments controlling the out-of-domain scoring run.
def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments with repository-relative defaults."""

    # Build a parser from this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__)
    # Allow pointing at a synthetic chat-eval CSV in tests.
    parser.add_argument(
        "--chat-eval-path",
        type=Path,
        default=_DEFAULT_CHAT_EVAL_PATH,
        help="Locked chat-style eval CSV (default: data/chat_eval/chat_style_eval_v1.csv).",
    )
    # Reports directory containing the frozen LSTM test_metrics.json.
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_DEFAULT_REPORTS_DIR,
        help="Directory with test_metrics.json (default: reports/lstm).",
    )
    # Checkpoint directory written by train_lstm.py.
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEFAULT_MODEL_DIR,
        help="Trained checkpoint directory (default: models/lstm).",
    )
    # Return the populated namespace for main().
    return parser.parse_args()


# Load the frozen operating point written after validation-only tuning.
def load_frozen_choices(reports_dir: Path) -> dict:
    """Return the LSTM test_metrics.json payload, failing if untrained."""

    # Resolve the TEST report that also stores chosen_threshold.
    metrics_path = reports_dir / "test_metrics.json"
    # Fail loudly rather than silently retuning or defaulting to 0.5.
    if not metrics_path.exists():
        # Tell the operator to freeze validation choices first.
        raise FileNotFoundError(
            f"No frozen LSTM report at {metrics_path}. "
            "Run scripts/train_lstm.py first; this script must not retune "
            "on chat_style_eval_v1.csv."
        )
    # Parse the JSON report.
    payload = json.loads(metrics_path.read_text())
    # Require the frozen threshold so a partial report cannot be reused silently.
    if "chosen_threshold" not in payload:
        # Ask for a regenerated report from the 70/20/10 protocol.
        raise ValueError(
            f"{metrics_path} is missing chosen_threshold. "
            "Re-run scripts/train_lstm.py with the 70/20/10 protocol."
        )
    # Return the full payload so split seeds can be reconstructed in the JSON stamp.
    return payload


# Load the TRAIN-fitted checkpoint and score the locked eval set only.
def main() -> None:
    """Score the locked chat-style eval set with a frozen word-BiLSTM checkpoint."""

    # Parse CLI arguments.
    args = parse_args()
    # Load the frozen threshold chosen on validation (never on the 200 rows).
    frozen = load_frozen_choices(args.reports_dir)
    # Read the frozen decision threshold; do not search it on chat eval.
    chosen_threshold = float(frozen["chosen_threshold"])
    # Read max_tokens so truncation matches the reported model.
    max_tokens = int(frozen.get("max_tokens", DEFAULT_MAX_TOKENS))
    # Read eval batch size from the recorded hyperparameters when present.
    hyper = frozen.get("hyperparameters") or {}
    # Fall back to the documented default eval batch.
    eval_batch_size = int(hyper.get("eval_batch_size", DEFAULT_EVAL_BATCH_SIZE))
    # Reconstruct device the same way training did (fp32 LSTM).
    device, _used_fp16, _reason = resolve_training_device()

    # Load tokenizer vocab + classifier + URL scaler from the local checkpoint.
    model, token_to_id, url_scaler, _hyperparams, saved_threshold = load_saved_classifier(
        args.model_dir
    )
    # Prefer the TEST-report threshold; the sidecar should match.
    threshold = chosen_threshold if chosen_threshold else saved_threshold
    # Load the locked eval set; predict_scam_proba never calls train_model.
    chat_eval_df = load_chat_style_eval_set(args.chat_eval_path)
    # Score with the frozen threshold; do not retune on these rows.
    chat_proba = predict_scam_proba(
        model,
        chat_eval_df["text"],
        token_to_id,
        url_scaler,
        max_tokens=max_tokens,
        batch_size=eval_batch_size,
        device=device,
    )
    # Build the metrics bundle; train_rows is the TRAIN size recorded at fit time.
    result = evaluate_from_proba(
        chat_eval_df["label"],
        chat_proba,
        threshold=threshold,
        train_rows=int(frozen.get("train_rows", 0)),
        source_frame=chat_eval_df,
    )

    # Ensure the reports directory exists before writing the OOD artifact.
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    # Persist the out-of-domain metrics next to the in-domain TEST report.
    metrics_path = args.reports_dir / "chat_style_eval_metrics.json"
    # Stamp processed_dir from the training report when present.
    processed_dir = Path(frozen.get("processed_dir", f"data/{_DEFAULT_PROCESSED_NAME}"))
    # Record that the 200-row file was never used for fitting or threshold search.
    metrics_path.write_text(
        json.dumps(
            {
                "trained_on_rows": int(frozen.get("train_rows", 0)),
                "trained_on": frozen.get("fitted_on", "train_only"),
                "processed_dir": str(processed_dir),
                "rewrite_method": frozen.get(
                    "rewrite_method", infer_rewrite_method(processed_dir)
                ),
                "held_out_in_domain_test_rows": int(frozen.get("test_rows", 0)),
                "chat_eval_rows": result.test_rows,
                "chosen_threshold": threshold,
                "threshold_source": "reports/lstm/test_metrics.json (validation-frozen)",
                "retuned_on_chat_eval": False,
                "chat_style_eval_training_allowed": False,
                "classification_report": _jsonify(result.classification_report),
                "confusion_matrix": result.confusion_matrix,
                "confusion_matrix_labels": ["legitimate", "scam"],
                "url_features": True,
                "live_url_reputation": False,
                "onnx_exported": False,
                "frontend_wired": False,
                "model_dir": str(args.model_dir),
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
        f"Evaluated on {result.test_rows} locked chat-eval rows "
        f"(threshold={threshold})"
    )
    # Show scam-class OOD metrics.
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
