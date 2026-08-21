"""CLI: fine-tune DistilBERT on LLM chat-register text and write Slice 5 reports.

Fits on TRAIN only, tunes the decision threshold on VALIDATION only (max scam
recall subject to legitimate recall >= 0.85), scores TEST once, then scores
the locked 200-row chat-style eval set predict-only. Never reads
data/chat_eval/ for fitting or threshold search. Does not export ONNX.

Usage (from ml/):
    uv run python scripts/train_distilbert.py
    uv run python scripts/train_distilbert.py --processed-dir data/processed_chat_llm
"""

# Import argparse to expose documented hyperparameters a reviewer might vary.
import argparse

# Import json to persist metrics reports as checked-in artifacts.
import json

# Import os so tokenizers/wandb stay quiet on this WSL2 box.
import os

# Import Path for portable input/output locations.
from pathlib import Path

# Import torch only to record the installed CUDA/fp16 environment.
import torch

# Import transformers to record the library version next to torch.
import transformers

# Reuse the same corpus loaders and split as the TF-IDF baseline.
from secure_chat_ml.baseline import (
    DEFAULT_LEGIT_RECALL_FLOOR,
    DEFAULT_THRESHOLD_GRID,
    infer_rewrite_method,
    load_chat_style_eval_set,
    load_processed_corpora,
    save_confusion_matrix_plot,
    stratified_split,
)

# Import DistilBERT training/inference helpers (live Hub load happens here).
from secure_chat_ml.distilbert import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_TRAIN_BATCH_SIZE,
    DistilBertHyperparameters,
    count_truncated_texts,
    evaluate_from_proba,
    fine_tune,
    load_pretrained_classifier,
    predict_scam_proba,
    resolve_training_device,
    save_classifier,
    tune_threshold_on_validation,
)

# Default to the completed llm_intent_v1 rewrite; do not re-run the 71k job.
_DEFAULT_PROCESSED_DIR = Path("data/processed_chat_llm")

# Write DistilBERT artifacts beside, never over, reports/baseline_metrics.json.
_DEFAULT_REPORTS_DIR = Path("reports/distilbert")

# Checkpoints are gitignored under ml/models/ (see the repo .gitignore).
_DEFAULT_MODEL_DIR = Path("models/distilbert")

# Locked evaluation-only file; never passed to fine_tune().
_DEFAULT_CHAT_EVAL_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")

# Published TF-IDF TEST report used only for the comparison JSON (read, not written).
_BASELINE_TEST_METRICS = Path("reports/baseline_metrics.json")

# Published TF-IDF chat-eval report used only for the comparison JSON.
_BASELINE_CHAT_METRICS = Path("reports/chat_style_eval_metrics.json")


# Keep HuggingFace tokenizers from spawning extra threads that warn under WSL2.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Disable wandb prompts if an operator has it installed globally.
os.environ.setdefault("WANDB_DISABLED", "true")


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


# Parse command-line arguments controlling the DistilBERT run.
def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments with the documented from-ml/ defaults."""

    # Build a parser from this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to the completed LLM rewrite; never data/chat_eval.
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=_DEFAULT_PROCESSED_DIR,
        help="Training-text directory (default: data/processed_chat_llm).",
    )
    # Keep DistilBERT reports in their own folder so TF-IDF JSON is never overwritten.
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_DEFAULT_REPORTS_DIR,
        help="Directory for DistilBERT JSON/PNG reports (default: reports/distilbert).",
    )
    # Save HuggingFace weights under the gitignored ml/models/ tree.
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEFAULT_MODEL_DIR,
        help="Directory to write the fine-tuned checkpoint (default: models/distilbert).",
    )
    # Allow tests to point at a synthetic chat-eval CSV.
    parser.add_argument(
        "--chat-eval-path",
        type=Path,
        default=_DEFAULT_CHAT_EVAL_PATH,
        help="Locked chat-style eval CSV (default: data/chat_eval/chat_style_eval_v1.csv).",
    )
    # Stratified train fraction matching train_baseline.py.
    parser.add_argument("--train-size", type=float, default=0.7, help="Train fraction.")
    # Stratified validation fraction used only for threshold selection.
    parser.add_argument("--val-size", type=float, default=0.2, help="Val fraction.")
    # Stratified test fraction scored once after freezing the threshold.
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction.")
    # Seed controlling the nested stratified split AND torch RNGs.
    parser.add_argument("--random-state", type=int, default=42, help="Split and torch seed.")
    # Documented Hub id; tests never invoke this script against the real Hub.
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model id or local directory (default: distilbert-base-uncased).",
    )
    # Documented truncation length (not searched on TEST).
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    # Documented train batch size; OOM retries step this down.
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    # Documented eval batch size.
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    # Documented AdamW learning rate (not searched on TEST).
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    # Documented epoch count (not searched on TEST).
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    # Optional skip so an operator can train without the 200-row file present.
    parser.add_argument(
        "--skip-chat-eval",
        action="store_true",
        help="Skip predict-only scoring of the locked chat-style eval set.",
    )
    # Return the populated namespace for main().
    return parser.parse_args()


# Build the hyperparameter bundle from CLI values.
def _hyperparams_from_args(
    args: argparse.Namespace, train_batch_size: int
) -> DistilBertHyperparameters:
    """Return frozen hyperparameters, using the batch size that actually ran."""

    # Record every documented knob so the JSON report is auditable.
    return DistilBertHyperparameters(
        model_name=args.model_name,
        max_length=args.max_length,
        train_batch_size=train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        seed=args.random_state,
    )


# Write a small JSON file describing torch/CUDA/fp16 on this machine.
def _training_env_payload(fp16_reason: str, used_fp16: bool) -> dict:
    """Return the environment block stored next to DistilBERT metrics."""

    # Record library versions so a reviewer can reproduce the pin.
    payload = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "fp16": bool(used_fp16),
        "fp16_reason": fp16_reason,
        "live_url_reputation": False,
        "onnx_exported": False,
        "frontend_wired": False,
    }
    # Add the GPU name when CUDA is visible (RTX 4060 Laptop GPU here).
    if torch.cuda.is_available():
        # Index 0 is the only device on this WSL2 box.
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        # Record total VRAM in bytes for the 8 GB capacity note.
        payload["cuda_device_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )
    # Return the JSON-ready dict.
    return payload


# Persist VAL metrics including the frozen threshold and why it was chosen.
def _val_metrics_payload(
    tuning,
    args: argparse.Namespace,
    hyperparams: DistilBertHyperparameters,
    fp16_reason: str,
    used_fp16: bool,
) -> dict:
    """Return the JSON body written to reports/distilbert/val_metrics.json."""

    # Include the chosen operating point, why it was chosen, and VAL metrics.
    return {
        "selection_rule": tuning.selection_rule,
        "selection_reason": tuning.selection_reason,
        "legit_recall_floor": tuning.legit_recall_floor,
        "floor_feasible": tuning.floor_feasible,
        "chosen_threshold": tuning.threshold,
        "grid_thresholds": tuning.grid_thresholds,
        "val_rows": tuning.val_rows,
        "classification_report": _jsonify(tuning.classification_report),
        "confusion_matrix": tuning.confusion_matrix,
        "confusion_matrix_labels": ["legitimate", "scam"],
        "random_state": args.random_state,
        "hyperparameters": {
            "model_name": hyperparams.model_name,
            "max_length": hyperparams.max_length,
            "train_batch_size": hyperparams.train_batch_size,
            "eval_batch_size": hyperparams.eval_batch_size,
            "learning_rate": hyperparams.learning_rate,
            "num_train_epochs": hyperparams.num_train_epochs,
            "warmup_ratio": hyperparams.warmup_ratio,
            "weight_decay": hyperparams.weight_decay,
            "seed": hyperparams.seed,
        },
        "training_env": _training_env_payload(fp16_reason, used_fp16),
        "chat_style_eval_used_for_tuning": False,
        "live_url_reputation": False,
    }


# Try the requested batch size, then 8, then 4 if CUDA runs out of memory.
def _fine_tune_with_oom_retry(
    model,
    tokenizer,
    train_texts,
    train_labels,
    args: argparse.Namespace,
    scratch_dir: Path,
) -> tuple[object, DistilBertHyperparameters]:
    """Fine-tune, stepping the batch size down on CUDA OOM only."""

    # Build the retry ladder from the requested size down, without duplicates.
    candidates: list[int] = []
    # Always try the operator-requested size first.
    for size in (args.batch_size, 8, 4):
        # Skip non-positive or already-queued sizes.
        if size >= 1 and size not in candidates:
            # Queue this batch size as a fallback.
            candidates.append(size)
    # Remember the last CUDA OOM so we can re-raise if every size fails.
    last_error: RuntimeError | None = None
    # Try each batch size until one completes.
    for batch_size in candidates:
        # Build hyperparameters at this batch size.
        hyperparams = _hyperparams_from_args(args, train_batch_size=batch_size)
        try:
            # Fit on TRAIN only at this batch size.
            trained = fine_tune(
                model,
                tokenizer,
                train_texts,
                train_labels,
                output_dir=scratch_dir,
                hyperparams=hyperparams,
            )
            # Announce when we had to step down from the requested size.
            if batch_size != args.batch_size:
                # Keep the OOM fallback visible in the training log.
                print(f"Resumed training at batch_size={batch_size} after CUDA OOM.")
            # Return the trained model and the knobs that actually ran.
            return trained, hyperparams
        except torch.cuda.OutOfMemoryError as exc:
            # Free fragmented CUDA memory before the next, smaller batch.
            torch.cuda.empty_cache()
            # Reload a fresh pretrained head because the OOM may have corrupted state.
            model, tokenizer = load_pretrained_classifier(args.model_name)
            # Remember the error in case the smallest batch still fails.
            last_error = exc
            # Tell the operator we are retrying smaller.
            print(f"CUDA OOM at batch_size={batch_size}; retrying smaller.")
            # Continue the fallback ladder.
            continue
    # Every batch size failed; surface the last OOM.
    assert last_error is not None
    # Re-raise so the operator sees the original CUDA message.
    raise last_error


# Optionally load published TF-IDF numbers for a side-by-side comparison file.
def _load_json_if_exists(path: Path) -> dict | None:
    """Return parsed JSON or None when the TF-IDF report is missing."""

    # Skip comparison cells when the baseline has not been trained in this tree.
    if not path.exists():
        # DistilBERT reports still stand on their own.
        return None
    # Parse the existing published report (never overwrite it).
    return json.loads(path.read_text())


# Extract per-class P/R/F1 plus confusion-matrix cells from a metrics payload.
def _summary_from_metrics(payload: dict) -> dict:
    """Return a compact P/R/F1 + CM summary for the comparison JSON."""

    # Read the sklearn classification_report block.
    report = payload["classification_report"]
    # Read the 2x2 matrix in [[TN, FP], [FN, TP]] order.
    matrix = payload["confusion_matrix"]
    # Bundle the cells the README tables quote.
    return {
        "legitimate_precision": report["legitimate"]["precision"],
        "legitimate_recall": report["legitimate"]["recall"],
        "legitimate_f1": report["legitimate"]["f1-score"],
        "scam_precision": report["scam"]["precision"],
        "scam_recall": report["scam"]["recall"],
        "scam_f1": report["scam"]["f1-score"],
        "confusion_matrix": matrix,
        "ham_warned": matrix[0][1],
        "scams_missed": matrix[1][0],
        "chosen_threshold": payload.get("chosen_threshold"),
    }


# Run load → split → TRAIN fine-tune → VAL threshold → TEST once → chat eval.
def main() -> None:
    """Fine-tune DistilBERT once and write TEST, VAL, and chat-eval reports."""

    # Parse CLI arguments (defaults match the documented from-ml/ command).
    args = parse_args()
    # Resolve CUDA vs CPU and whether fp16 will actually run.
    device, used_fp16, fp16_reason = resolve_training_device()
    # Print the device story before the Hub download so OOM notes are in context.
    print(f"Device: {device}  fp16: {used_fp16} ({fp16_reason})")

    # Load every rewritten corpus into one combined dataset (refuses chat_eval/).
    combined = load_processed_corpora(args.processed_dir)
    # Split with the same 70/20/10 seed as train_baseline.py.
    train_df, val_df, test_df = stratified_split(
        combined,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    # Announce split sizes so a reviewer can confirm 49,958 / 14,275 / 7,137.
    print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}  Test rows: {len(test_df)}")

    # Download (or reuse the HF cache of) DistilBERT-base plus its tokenizer.
    model, tokenizer = load_pretrained_classifier(args.model_name)
    # Count TRAIN rows truncated at max_length for the README honesty note.
    truncated_train = count_truncated_texts(
        tokenizer, train_df["text"].astype(str).tolist(), args.max_length
    )
    # Print truncation so it is visible even if JSON is not opened.
    print(f"TRAIN rows truncated at max_length={args.max_length}: {truncated_train}")

    # Scratch directory for Trainer logs; the real checkpoint is --model-dir.
    scratch_dir = args.model_dir / "_trainer_scratch"
    # Fine-tune on TRAIN only, stepping the batch size down on CUDA OOM.
    model, hyperparams = _fine_tune_with_oom_retry(
        model,
        tokenizer,
        train_df["text"],
        train_df["label"],
        args,
        scratch_dir,
    )

    # Score VAL probabilities with the TRAIN-fitted model (no VAL gradient steps).
    val_proba = predict_scam_proba(
        model,
        tokenizer,
        val_df["text"],
        max_length=hyperparams.max_length,
        batch_size=hyperparams.eval_batch_size,
        device=device,
        use_fp16=used_fp16,
    )
    # Freeze the threshold on VAL using the same rule as the TF-IDF baseline.
    tuning = tune_threshold_on_validation(
        val_df["label"],
        val_proba,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        legit_recall_floor=DEFAULT_LEGIT_RECALL_FLOOR,
    )
    # Show the frozen operating point before TEST is touched.
    print(
        f"Frozen threshold: {tuning.threshold}  Reason: {tuning.selection_reason}  "
        f"floor_feasible={tuning.floor_feasible}"
    )

    # Score TEST once with the frozen threshold; do not search it here.
    test_proba = predict_scam_proba(
        model,
        tokenizer,
        test_df["text"],
        max_length=hyperparams.max_length,
        batch_size=hyperparams.eval_batch_size,
        device=device,
        use_fp16=used_fp16,
    )
    # Build the TEST metrics bundle quoted by the README.
    test_eval = evaluate_from_proba(
        test_df["label"],
        test_proba,
        threshold=tuning.threshold,
        train_rows=len(train_df),
        source_frame=test_df,
        val_rows=len(val_df),
    )

    # Persist the fine-tuned weights + tokenizer under the gitignored models/ tree.
    save_classifier(model, tokenizer, args.model_dir)
    # Ensure the DistilBERT reports directory exists before writing artifacts.
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # Persist TEST metrics; never write reports/baseline_metrics.json.
    test_path = args.reports_dir / "test_metrics.json"
    # Include split sizes, frozen threshold, and documented hyperparameters.
    test_payload = {
        "train_rows": test_eval.train_rows,
        "val_rows": len(val_df),
        "test_rows": test_eval.test_rows,
        "chosen_threshold": tuning.threshold,
        "selection_reason": tuning.selection_reason,
        "selection_rule": tuning.selection_rule,
        "legit_recall_floor": tuning.legit_recall_floor,
        "floor_feasible": tuning.floor_feasible,
        "classification_report": _jsonify(test_eval.classification_report),
        "confusion_matrix": test_eval.confusion_matrix,
        "confusion_matrix_labels": ["legitimate", "scam"],
        "source_counts_in_test_split": test_eval.source_counts,
        "random_state": args.random_state,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "processed_dir": str(args.processed_dir),
        "rewrite_method": infer_rewrite_method(args.processed_dir),
        "truncated_train_rows": truncated_train,
        "max_length": hyperparams.max_length,
        "hyperparameters": {
            "model_name": hyperparams.model_name,
            "max_length": hyperparams.max_length,
            "train_batch_size": hyperparams.train_batch_size,
            "eval_batch_size": hyperparams.eval_batch_size,
            "learning_rate": hyperparams.learning_rate,
            "num_train_epochs": hyperparams.num_train_epochs,
            "warmup_ratio": hyperparams.warmup_ratio,
            "weight_decay": hyperparams.weight_decay,
            "seed": hyperparams.seed,
        },
        "training_env": _training_env_payload(fp16_reason, used_fp16),
        "model_dir": str(args.model_dir),
        "url_features": False,
        "live_url_reputation": False,
        "onnx_exported": False,
        "chat_style_eval_used_for_training": False,
        "chat_style_eval_used_for_tuning": False,
        "fitted_on": "train_only",
    }
    # Write TEST JSON with stable indentation for git diffs.
    test_path.write_text(json.dumps(test_payload, indent=2))
    # Persist VAL metrics separately so the threshold search remains auditable.
    val_path = args.reports_dir / "val_metrics.json"
    # Write the validation payload next to the TEST report.
    val_path.write_text(
        json.dumps(
            _val_metrics_payload(tuning, args, hyperparams, fp16_reason, used_fp16),
            indent=2,
        )
    )
    # Save the TEST confusion-matrix heatmap with a DistilBERT title.
    cm_path = args.reports_dir / "confusion_matrix.png"
    # Reuse the baseline plot helper with a DistilBERT-specific title.
    save_confusion_matrix_plot(
        test_eval,
        cm_path,
        title="DistilBERT-base (test, TRAIN-only fine-tune)",
    )

    # Print TEST scam metrics — these are the numbers the README must quote.
    scam_metrics = test_eval.classification_report["scam"]
    # Show TEST scam P/R/F1.
    print(
        "TEST scam class -> precision: "
        f"{scam_metrics['precision']:.3f}  recall: {scam_metrics['recall']:.3f}  "
        f"f1: {scam_metrics['f1-score']:.3f}"
    )
    # Show the TEST confusion matrix in [[TN, FP], [FN, TP]] order.
    print(f"TEST confusion matrix ([[TN, FP], [FN, TP]]): {test_eval.confusion_matrix}")
    # Point the operator at the written TEST artifacts.
    print(f"Wrote {test_path}, {val_path}, and {cm_path}")

    # Score the locked chat-eval set predict-only unless the operator skipped it.
    chat_path = args.reports_dir / "chat_style_eval_metrics.json"
    # Stay None when --skip-chat-eval so the comparison JSON can omit that block.
    chat_payload: dict | None = None
    # Honor --skip-chat-eval for operators who only want in-domain TEST.
    if not args.skip_chat_eval:
        # Load the locked 200-row file; this path is never concatenated into TRAIN.
        chat_eval_df = load_chat_style_eval_set(args.chat_eval_path)
        # Predict with the TRAIN-fitted checkpoint; do not fine-tune on these rows.
        chat_proba = predict_scam_proba(
            model,
            tokenizer,
            chat_eval_df["text"],
            max_length=hyperparams.max_length,
            batch_size=hyperparams.eval_batch_size,
            device=device,
            use_fp16=used_fp16,
        )
        # Apply the VAL-frozen threshold; do not retune on the 200 rows.
        chat_eval = evaluate_from_proba(
            chat_eval_df["label"],
            chat_proba,
            threshold=tuning.threshold,
            train_rows=len(train_df),
            source_frame=chat_eval_df,
        )
        # Record that the 200-row file was never used for fitting or threshold search.
        chat_payload = {
            "trained_on_rows": len(train_df),
            "trained_on": f"train_only_{args.processed_dir.name}",
            "processed_dir": str(args.processed_dir),
            "rewrite_method": infer_rewrite_method(args.processed_dir),
            "held_out_in_domain_test_rows": len(test_df),
            "held_out_in_domain_val_rows": len(val_df),
            "chat_eval_rows": chat_eval.test_rows,
            "chosen_threshold": tuning.threshold,
            "threshold_source": "reports/distilbert/val_metrics.json (validation-frozen)",
            "retuned_on_chat_eval": False,
            "chat_style_eval_training_allowed": False,
            "classification_report": _jsonify(chat_eval.classification_report),
            "confusion_matrix": chat_eval.confusion_matrix,
            "confusion_matrix_labels": ["legitimate", "scam"],
            "live_url_reputation": False,
            "onnx_exported": False,
            "model_dir": str(args.model_dir),
        }
        # Persist the out-of-domain metrics next to the in-domain TEST report.
        chat_path.write_text(json.dumps(chat_payload, indent=2))
        # Save a second confusion-matrix image for the locked chat set.
        chat_cm_path = args.reports_dir / "chat_eval_confusion_matrix.png"
        # Title the plot so it cannot be mistaken for the in-domain TEST figure.
        save_confusion_matrix_plot(
            chat_eval,
            chat_cm_path,
            title="DistilBERT-base (locked chat-style eval, predict-only)",
        )
        # Print chat-eval scam and ham metrics for the README comparison.
        chat_scam = chat_eval.classification_report["scam"]
        # Also print legitimate metrics so false-positive cost is visible.
        chat_legit = chat_eval.classification_report["legitimate"]
        # State clearly that chat eval was scored, not trained on.
        print(
            f"Chat eval: {chat_eval.test_rows} locked rows "
            f"(threshold={tuning.threshold}, fitted on {len(train_df)} TRAIN rows)"
        )
        # Show scam-class OOD metrics.
        print(
            "Chat eval scam  -> precision: "
            f"{chat_scam['precision']:.3f}  recall: {chat_scam['recall']:.3f}  "
            f"f1: {chat_scam['f1-score']:.3f}"
        )
        # Show legitimate-class OOD metrics.
        print(
            "Chat eval legit -> precision: "
            f"{chat_legit['precision']:.3f}  recall: {chat_legit['recall']:.3f}  "
            f"f1: {chat_legit['f1-score']:.3f}"
        )
        # Show the OOD confusion matrix in [[TN, FP], [FN, TP]] order.
        print(f"Chat eval confusion matrix ([[TN, FP], [FN, TP]]): {chat_eval.confusion_matrix}")
        # Point the operator at the written OOD artifacts.
        print(f"Wrote {chat_path} and {chat_cm_path}")

    # Write a compact DistilBERT-vs-TF-IDF comparison when baseline reports exist.
    baseline_test = _load_json_if_exists(_BASELINE_TEST_METRICS)
    # Load the published TF-IDF chat-eval numbers for the same 200 rows.
    baseline_chat = _load_json_if_exists(_BASELINE_CHAT_METRICS)
    # Only emit the comparison file when at least the TF-IDF TEST report is present.
    if baseline_test is not None:
        # Start with in-domain TEST for both models.
        comparison = {
            "tfidf_test": _summary_from_metrics(baseline_test),
            "distilbert_test": _summary_from_metrics(test_payload),
            "tfidf_fitted_on": "train_only for TEST; train_plus_val for chat eval",
            "distilbert_fitted_on": "train_only for TEST and chat eval",
            "same_split_seed": 42,
            "same_selection_rule": tuning.selection_rule,
        }
        # Attach chat-eval comparison when both sides exist.
        if baseline_chat is not None and chat_payload is not None:
            # Side-by-side locked 200-row numbers (predict-only on both tracks).
            comparison["tfidf_chat_eval"] = _summary_from_metrics(baseline_chat)
            # DistilBERT chat-eval used the TRAIN-only checkpoint.
            comparison["distilbert_chat_eval"] = _summary_from_metrics(chat_payload)
        # Persist the comparison next to the DistilBERT reports.
        comparison_path = args.reports_dir / "comparison_vs_tfidf.json"
        # Write JSON for the README tables to quote.
        comparison_path.write_text(json.dumps(_jsonify(comparison), indent=2))
        # Point the operator at the comparison artifact.
        print(f"Wrote {comparison_path}")


# Run the training pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
