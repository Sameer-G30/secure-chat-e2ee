"""CLI: train a word BiLSTM + URL-concat classifier and write reports/lstm/.

Fits vocab, URL StandardScaler, embeddings, BiLSTM, and the head on TRAIN
only. Tunes the decision threshold on VALIDATION only (max scam recall
subject to legitimate recall >= 0.85). Scores TEST once, then scores the
locked 200-row chat-style eval set predict-only. Never reads
data/chat_eval/ for fitting or threshold search. Does not export ONNX.
Does not train a character-level LSTM.

Usage (from ml/):
    uv run python scripts/train_lstm.py
    uv run python scripts/train_lstm.py --processed-dir data/processed_chat_llm
"""

# Import argparse to expose documented hyperparameters a reviewer might vary.
import argparse

# Import json to persist metrics reports as checked-in artifacts.
import json

# Import time so the report can record TRAIN wall-clock seconds.
import time

# Import Path for portable input/output locations.
from pathlib import Path

# Import torch only to record the installed CUDA/fp32 environment.
import torch

# Reuse the same corpus loaders and split as the TF-IDF / DistilBERT tracks.
from secure_chat_ml.baseline import (
    DEFAULT_LEGIT_RECALL_FLOOR,
    DEFAULT_THRESHOLD_GRID,
    infer_rewrite_method,
    load_chat_style_eval_set,
    load_processed_corpora,
    save_confusion_matrix_plot,
    stratified_split,
)
from secure_chat_ml.lstm import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_DIM,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_VOCAB_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_TRAIN_EPOCHS,
    LstmHyperparameters,
    analyze_link_errors,
    assert_not_chat_eval_path,
    build_model,
    build_vocab,
    count_truncated_texts,
    evaluate_from_proba,
    fit_url_scaler,
    predict_scam_proba,
    recommend_char_lstm_exploration,
    render_char_lstm_decision_markdown,
    resolve_training_device,
    save_classifier,
    train_model,
    tune_threshold_on_validation,
)
from secure_chat_ml.url_features import URL_FEATURE_NAMES

# Default to the completed llm_intent_v1 rewrite; do not re-run the 71k job.
_DEFAULT_PROCESSED_DIR = Path("data/processed_chat_llm")

# Write LSTM artifacts beside, never over, reports/baseline_metrics.json.
_DEFAULT_REPORTS_DIR = Path("reports/lstm")

# Checkpoints are gitignored under ml/models/ (see the repo .gitignore).
_DEFAULT_MODEL_DIR = Path("models/lstm")

# Locked evaluation-only file; never passed to train_model().
_DEFAULT_CHAT_EVAL_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")

# Published TF-IDF TEST report used only for the comparison JSON (read, not written).
_BASELINE_TEST_METRICS = Path("reports/baseline_metrics.json")

# Published TF-IDF chat-eval report used only for the comparison JSON.
_BASELINE_CHAT_METRICS = Path("reports/chat_style_eval_metrics.json")

# Published DistilBERT TEST report used only for the comparison JSON.
_DISTILBERT_TEST_METRICS = Path("reports/distilbert/test_metrics.json")

# Published DistilBERT chat-eval report used only for the comparison JSON.
_DISTILBERT_CHAT_METRICS = Path("reports/distilbert/chat_style_eval_metrics.json")


# Convert a numpy-ish value inside a classification_report into JSON-safe types.
def _jsonify(value: object) -> object:
    """Return a JSON-serializable copy of nested sklearn report structures."""

    # Recurse into dictionaries produced by classification_report(..., output_dict=True).
    if isinstance(value, dict):
        # Convert every nested value the same way.
        return {str(key): _jsonify(item) for key, item in value.items()}
    # Recurse into confusion-matrix rows and FN/FP lists.
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


# Parse command-line arguments controlling the word-BiLSTM run.
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
    # Keep LSTM reports in their own folder so TF-IDF/DistilBERT JSON is never overwritten.
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=_DEFAULT_REPORTS_DIR,
        help="Directory for LSTM JSON/PNG reports (default: reports/lstm).",
    )
    # Save weights under the gitignored ml/models/ tree.
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEFAULT_MODEL_DIR,
        help="Directory to write the trained checkpoint (default: models/lstm).",
    )
    # Allow tests to point at a synthetic chat-eval CSV.
    parser.add_argument(
        "--chat-eval-path",
        type=Path,
        default=_DEFAULT_CHAT_EVAL_PATH,
        help="Locked chat-style eval CSV (default: data/chat_eval/chat_style_eval_v1.csv).",
    )
    # Stratified train fraction matching train_baseline.py / train_distilbert.py.
    parser.add_argument("--train-size", type=float, default=0.7, help="Train fraction.")
    # Stratified validation fraction used only for threshold selection.
    parser.add_argument("--val-size", type=float, default=0.2, help="Val fraction.")
    # Stratified test fraction scored once after freezing the threshold.
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction.")
    # Seed controlling the nested stratified split AND torch RNGs.
    parser.add_argument("--random-state", type=int, default=42, help="Split and torch seed.")
    # Documented embedding width (not searched on TEST).
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    # Documented per-direction hidden size (not searched on TEST).
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    # Documented LSTM depth (not searched on TEST).
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    # Documented dropout on embeddings and the pooled vector.
    parser.add_argument("--dropout", type=float, default=0.3)
    # Documented pad/truncate length in word tokens.
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    # Documented TRAIN vocabulary cap.
    parser.add_argument("--max-vocab-size", type=int, default=DEFAULT_MAX_VOCAB_SIZE)
    # Documented train batch size.
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    # Documented eval batch size.
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    # Documented Adam learning rate (not searched on TEST).
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    # Documented epoch count (not searched on TEST).
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    # Optional explicit VAL threshold list; default remains 0.30 … 0.70.
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=None,
        help="VAL P(scam) thresholds to search (default: 0.30 0.35 ... 0.70).",
    )
    # Optional skip so an operator can train without the 200-row file present.
    parser.add_argument(
        "--skip-chat-eval",
        action="store_true",
        help="Skip predict-only scoring of the locked chat-style eval set.",
    )
    # Return the populated namespace for main().
    return parser.parse_args()


# Build the hyperparameter bundle from CLI values.
def _hyperparams_from_args(args: argparse.Namespace) -> LstmHyperparameters:
    """Return frozen hyperparameters recorded in JSON reports."""

    # Record every documented knob so the JSON report is auditable.
    return LstmHyperparameters(
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_tokens=args.max_tokens,
        max_vocab_size=args.max_vocab_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        seed=args.random_state,
    )


# Resolve the VAL threshold list from CLI flags without inventing a C grid.
def _resolve_threshold_grid(args: argparse.Namespace) -> tuple[float, ...]:
    """Return the VAL P(scam) grid actually searched for this run."""

    # An explicit list always wins so a reviewer can pass a custom grid.
    if args.threshold_grid is not None:
        # Keep order stable and drop accidental duplicates.
        unique: list[float] = []
        # Walk values in the order the operator typed them.
        for value in args.threshold_grid:
            # Cast through float so JSON later stores plain numbers.
            as_float = float(value)
            # Skip duplicates while preserving the first occurrence.
            if as_float not in unique:
                # Queue this cut for VAL search.
                unique.append(as_float)
        # Sort low-to-high so reports stay comparable across runs.
        return tuple(sorted(unique))
    # Documented default remains 0.30 ... 0.70; this LSTM has no C grid.
    return DEFAULT_THRESHOLD_GRID


# Write a small JSON file describing torch/CUDA/fp32 on this machine.
def _training_env_payload(fp16_reason: str, used_fp16: bool) -> dict:
    """Return the environment block stored next to LSTM metrics."""

    # Record library versions so a reviewer can reproduce the pin.
    payload = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "fp16": bool(used_fp16),
        "fp16_reason": fp16_reason,
        "url_features": True,
        "url_feature_count": len(URL_FEATURE_NAMES),
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


# Serialize hyperparameters into the JSON reports train/eval scripts share.
def _hyperparams_payload(hyperparams: LstmHyperparameters, vocab_size: int) -> dict:
    """Return the hyperparameters object written into VAL/TEST JSON."""

    # Include architecture notes so pooling/tokenizer are not implied.
    return {
        "embed_dim": hyperparams.embed_dim,
        "hidden_size": hyperparams.hidden_size,
        "num_layers": hyperparams.num_layers,
        "dropout": hyperparams.dropout,
        "max_tokens": hyperparams.max_tokens,
        "max_vocab_size": hyperparams.max_vocab_size,
        "vocab_size": int(vocab_size),
        "batch_size": hyperparams.batch_size,
        "eval_batch_size": hyperparams.eval_batch_size,
        "learning_rate": hyperparams.learning_rate,
        "num_train_epochs": hyperparams.num_train_epochs,
        "grad_clip": hyperparams.grad_clip,
        "seed": hyperparams.seed,
        "pooling": hyperparams.pooling,
        "tokenizer": hyperparams.tokenizer,
        "url_feature_concat": True,
        "url_feature_count": len(URL_FEATURE_NAMES),
    }


# Persist VAL metrics including the frozen threshold and why it was chosen.
def _val_metrics_payload(
    tuning,
    args: argparse.Namespace,
    hyperparams: LstmHyperparameters,
    vocab_size: int,
    fp16_reason: str,
    used_fp16: bool,
) -> dict:
    """Return the JSON body written to reports/lstm/val_metrics.json."""

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
        "hyperparameters": _hyperparams_payload(hyperparams, vocab_size),
        "training_env": _training_env_payload(fp16_reason, used_fp16),
        "url_features": True,
        "live_url_reputation": False,
        "onnx_exported": False,
        "frontend_wired": False,
        "fitted_on": "train_only",
        "chat_style_eval_used_for_tuning": False,
    }


# Optionally load published TF-IDF/DistilBERT numbers for a side-by-side file.
def _load_json_if_exists(path: Path) -> dict | None:
    """Return parsed JSON or None when the comparison report is missing."""

    # Skip comparison cells when the other track has not been trained in this tree.
    if not path.exists():
        # LSTM reports still stand on their own.
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


# Count how many LSTM FNs would be "extra" vs a published FN count and URL-related.
def _extra_fn_url_related(analysis: dict, published_fn: int) -> tuple[int, int]:
    """Return (extra_fn, extra_fn_url_related) using published FN as the baseline.

    Without the other model's per-row predictions we cannot intersect FN sets.
    Extra FN is max(lstm_fn - published_fn, 0). URL-related extra is min of
    that extra and the LSTM URL-related FN count (upper bound on overlap).
    """

    # LSTM false-negative rows already scored with on-device URL flags.
    fn_rows = analysis["false_negatives"]
    # LSTM FN count on this split.
    lstm_fn = len(fn_rows)
    # Extra misses versus the published model on the same split size.
    extra = max(lstm_fn - int(published_fn), 0)
    # URL-related LSTM FNs (has_url or any lexical phishing flag).
    url_related = int(analysis["fn_url_related"])
    # Extra URL-related cannot exceed extra or the URL-related LSTM FN count.
    extra_url = min(extra, url_related)
    # Return both counts for criterion B.
    return extra, extra_url


# Run load → split → TRAIN fit → VAL threshold → TEST once → chat eval.
def main() -> None:
    """Train the word BiLSTM once and write TEST, VAL, and chat-eval reports."""

    # Parse CLI arguments (defaults match the documented from-ml/ command).
    args = parse_args()
    # Refuse a mis-pointed chat_eval training directory before any I/O.
    assert_not_chat_eval_path(args.processed_dir)
    # Freeze the VAL threshold list before any probabilities are scored.
    threshold_grid = _resolve_threshold_grid(args)
    # Resolve CUDA vs CPU; LSTM always reports fp16=False.
    device, used_fp16, fp16_reason = resolve_training_device()
    # Print the device story before encoding 50k rows.
    print(f"Device: {device}  fp16: {used_fp16} ({fp16_reason})")

    # Load every rewritten corpus into one combined dataset (refuses chat_eval/).
    combined = load_processed_corpora(args.processed_dir)
    # Split with the same 70/20/10 seed as train_baseline.py / train_distilbert.py.
    train_df, val_df, test_df = stratified_split(
        combined,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    # Announce split sizes so a reviewer can confirm 49,958 / 14,275 / 7,137.
    print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}  Test rows: {len(test_df)}")

    # Bundle documented knobs (not searched on TEST or chat_eval).
    hyperparams = _hyperparams_from_args(args)
    # Build the TRAIN-only vocabulary; VAL/TEST tokens may become UNK.
    token_to_id = build_vocab(train_df["text"], max_vocab_size=hyperparams.max_vocab_size)
    # Record the actual vocab size (PAD/UNK plus kept TRAIN tokens).
    vocab_size = len(token_to_id)
    # Count TRAIN rows truncated at max_tokens for the README honesty note.
    truncated_train = count_truncated_texts(train_df["text"], hyperparams.max_tokens)
    # Print vocab and truncation so they are visible even if JSON is not opened.
    print(
        f"TRAIN vocab_size={vocab_size}  rows truncated at max_tokens="
        f"{hyperparams.max_tokens}: {truncated_train}"
    )
    # Fit the URL StandardScaler on TRAIN only (zeros for link-free DMs).
    url_scaler = fit_url_scaler(train_df["text"])
    # Construct an untrained network whose head width includes URL features.
    model = build_model(vocab_size=vocab_size, hyperparams=hyperparams)

    # Time the TRAIN loop for the report (word BiLSTM should be far faster than DistilBERT).
    train_started = time.perf_counter()
    # Fit embeddings, BiLSTM, and the head on TRAIN only.
    model = train_model(
        model,
        train_df["text"],
        train_df["label"],
        token_to_id,
        url_scaler,
        hyperparams=hyperparams,
    )
    # Elapsed TRAIN seconds (excludes VAL/TEST scoring).
    train_seconds = time.perf_counter() - train_started
    # Show TRAIN wall-clock before threshold search.
    print(f"TRAIN wall-clock seconds: {train_seconds:.1f}")

    # Score VAL probabilities with the TRAIN-fitted model (no VAL gradient steps).
    val_proba = predict_scam_proba(
        model,
        val_df["text"],
        token_to_id,
        url_scaler,
        max_tokens=hyperparams.max_tokens,
        batch_size=hyperparams.eval_batch_size,
        device=device,
    )
    # Freeze the threshold on VAL using the same rule as TF-IDF / DistilBERT.
    tuning = tune_threshold_on_validation(
        val_df["label"],
        val_proba,
        threshold_grid=threshold_grid,
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
        test_df["text"],
        token_to_id,
        url_scaler,
        max_tokens=hyperparams.max_tokens,
        batch_size=hyperparams.eval_batch_size,
        device=device,
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

    # Persist the trained weights + TRAIN vocab/scaler under the gitignored models/ tree.
    save_classifier(
        model,
        token_to_id,
        url_scaler,
        args.model_dir,
        hyperparams=hyperparams,
        threshold=tuning.threshold,
    )
    # Ensure the LSTM reports directory exists before writing artifacts.
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # Persist TEST metrics; never write reports/baseline_metrics.json or distilbert/.
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
        "max_tokens": hyperparams.max_tokens,
        "vocab_size": vocab_size,
        "train_wall_clock_seconds": train_seconds,
        "hyperparameters": _hyperparams_payload(hyperparams, vocab_size),
        "training_env": _training_env_payload(fp16_reason, used_fp16),
        "model_dir": str(args.model_dir),
        "url_features": True,
        "live_url_reputation": False,
        "onnx_exported": False,
        "frontend_wired": False,
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
            _val_metrics_payload(
                tuning, args, hyperparams, vocab_size, fp16_reason, used_fp16
            ),
            indent=2,
        )
    )
    # Save the TEST confusion-matrix heatmap with a word-BiLSTM title.
    cm_path = args.reports_dir / "confusion_matrix.png"
    # Reuse the baseline plot helper with an LSTM-specific title.
    save_confusion_matrix_plot(
        test_eval,
        cm_path,
        title="Word BiLSTM + URL concat (test, TRAIN-only fit)",
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

    # Analyze TEST FN/FP URL flags after the frozen threshold (no refit).
    test_link = analyze_link_errors(
        test_df["text"],
        test_df["label"],
        test_proba,
        threshold=tuning.threshold,
        split_name="test",
    )

    # Score the locked chat-eval set predict-only unless the operator skipped it.
    chat_path = args.reports_dir / "chat_style_eval_metrics.json"
    # Stay None when --skip-chat-eval so the comparison JSON can omit that block.
    chat_payload: dict | None = None
    # Stay None when chat eval is skipped so link analysis can omit that split.
    chat_link: dict | None = None
    # Honor --skip-chat-eval for operators who only want in-domain TEST.
    if not args.skip_chat_eval:
        # Load the locked 200-row file; this path is never concatenated into TRAIN.
        chat_eval_df = load_chat_style_eval_set(args.chat_eval_path)
        # Predict with the TRAIN-fitted checkpoint; do not train on these rows.
        chat_proba = predict_scam_proba(
            model,
            chat_eval_df["text"],
            token_to_id,
            url_scaler,
            max_tokens=hyperparams.max_tokens,
            batch_size=hyperparams.eval_batch_size,
            device=device,
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
            "threshold_source": f"{val_path.as_posix()} (validation-frozen)",
            "retuned_on_chat_eval": False,
            "chat_style_eval_training_allowed": False,
            "classification_report": _jsonify(chat_eval.classification_report),
            "confusion_matrix": chat_eval.confusion_matrix,
            "confusion_matrix_labels": ["legitimate", "scam"],
            "url_features": True,
            "live_url_reputation": False,
            "onnx_exported": False,
            "frontend_wired": False,
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
            title="Word BiLSTM + URL concat (locked chat-style eval, predict-only)",
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
        print(
            f"Chat eval confusion matrix ([[TN, FP], [FN, TP]]): {chat_eval.confusion_matrix}"
        )
        # Point the operator at the written OOD artifacts.
        print(f"Wrote {chat_path} and {chat_cm_path}")
        # Analyze chat-eval FN/FP URL flags after the frozen threshold.
        chat_link = analyze_link_errors(
            chat_eval_df["text"],
            chat_eval_df["label"],
            chat_proba,
            threshold=tuning.threshold,
            split_name="chat_style_eval",
        )

    # Load published TF-IDF / DistilBERT numbers when present (never retrain them).
    baseline_test = _load_json_if_exists(_BASELINE_TEST_METRICS)
    # Load the published TF-IDF chat-eval numbers for the same 200 rows.
    baseline_chat = _load_json_if_exists(_BASELINE_CHAT_METRICS)
    # Load DistilBERT TEST metrics for the comparison JSON.
    distilbert_test = _load_json_if_exists(_DISTILBERT_TEST_METRICS)
    # Load DistilBERT chat-eval metrics for the comparison JSON.
    distilbert_chat = _load_json_if_exists(_DISTILBERT_CHAT_METRICS)

    # Write a three-way comparison when at least the TF-IDF TEST report is present.
    if baseline_test is not None:
        # Start with in-domain TEST for TF-IDF and LSTM.
        comparison = {
            "tfidf_test": _summary_from_metrics(baseline_test),
            "lstm_test": _summary_from_metrics(test_payload),
            "tfidf_fitted_on": "train_only for TEST; train_plus_val for chat eval",
            "distilbert_fitted_on": "train_only for TEST and chat eval",
            "lstm_fitted_on": "train_only for TEST and chat eval",
            "same_split_seed": 42,
            "same_selection_rule": tuning.selection_rule,
        }
        # Attach DistilBERT TEST when that published report exists.
        if distilbert_test is not None:
            # Side-by-side in-domain TEST for all three models.
            comparison["distilbert_test"] = _summary_from_metrics(distilbert_test)
        # Attach chat-eval comparison when LSTM scored the 200 rows.
        if baseline_chat is not None and chat_payload is not None:
            # Side-by-side locked 200-row numbers (predict-only on LSTM/DistilBERT).
            comparison["tfidf_chat_eval"] = _summary_from_metrics(baseline_chat)
            # LSTM chat-eval used the TRAIN-only checkpoint.
            comparison["lstm_chat_eval"] = _summary_from_metrics(chat_payload)
            # Attach DistilBERT chat-eval when present.
            if distilbert_chat is not None:
                # Third column for the README table.
                comparison["distilbert_chat_eval"] = _summary_from_metrics(distilbert_chat)
        # Persist the comparison next to the LSTM reports (never over DistilBERT's file).
        comparison_path = args.reports_dir / "comparison_vs_tfidf_and_distilbert.json"
        # Write JSON for the README tables to quote.
        comparison_path.write_text(json.dumps(_jsonify(comparison), indent=2))
        # Point the operator at the comparison artifact.
        print(f"Wrote {comparison_path}")

    # Default published FN counts when comparison JSON is missing.
    tfidf_test_fn = 27
    # DistilBERT published TEST misses.
    distilbert_test_fn = 60
    # TF-IDF published chat-eval misses.
    tfidf_chat_fn = 0
    # TF-IDF TEST scam recall from the published report (fallback to known value).
    tfidf_test_recall = 0.9922279792746114
    # DistilBERT TEST scam recall from the published report.
    distilbert_test_recall = 0.9827288428324698
    # TF-IDF chat-eval scam recall.
    tfidf_chat_recall = 1.0
    # TF-IDF chat ham warned.
    tfidf_chat_fp = 70
    # Override defaults from on-disk published reports when they exist.
    if baseline_test is not None:
        # Read TEST missed-scam cell [[TN, FP], [FN, TP]].
        tfidf_test_fn = int(baseline_test["confusion_matrix"][1][0])
        # Read TEST scam recall.
        tfidf_test_recall = float(baseline_test["classification_report"]["scam"]["recall"])
    # Override DistilBERT TEST numbers when the Slice 5 report exists.
    if distilbert_test is not None:
        # Read DistilBERT TEST FN count.
        distilbert_test_fn = int(distilbert_test["confusion_matrix"][1][0])
        # Read DistilBERT TEST scam recall.
        distilbert_test_recall = float(
            distilbert_test["classification_report"]["scam"]["recall"]
        )
    # Override TF-IDF chat-eval numbers when the published OOD report exists.
    if baseline_chat is not None:
        # Read chat-eval FN count (published TF-IDF is 0).
        tfidf_chat_fn = int(baseline_chat["confusion_matrix"][1][0])
        # Read chat-eval scam recall.
        tfidf_chat_recall = float(baseline_chat["classification_report"]["scam"]["recall"])
        # Read chat-eval ham warned.
        tfidf_chat_fp = int(baseline_chat["confusion_matrix"][0][1])

    # LSTM TEST confusion-matrix cells.
    lstm_test_fn = int(test_eval.confusion_matrix[1][0])
    # LSTM TEST scam recall.
    lstm_test_recall = float(test_eval.classification_report["scam"]["recall"])
    # Chat-eval LSTM metrics default to zeros when --skip-chat-eval.
    lstm_chat_fn = 0
    # Chat-eval LSTM ham warned.
    lstm_chat_fp = 0
    # Chat-eval LSTM scam recall.
    lstm_chat_recall = 0.0
    # Fill chat-eval LSTM numbers when that split was scored.
    if chat_payload is not None:
        # Read LSTM chat-eval FN count.
        lstm_chat_fn = int(chat_payload["confusion_matrix"][1][0])
        # Read LSTM chat-eval FP count.
        lstm_chat_fp = int(chat_payload["confusion_matrix"][0][1])
        # Read LSTM chat-eval scam recall.
        lstm_chat_recall = float(chat_payload["classification_report"]["scam"]["recall"])

    # Extra chat-eval FNs vs TF-IDF that are URL-related (upper bound without TF-IDF rows).
    extra_chat, extra_chat_url = (0, 0)
    # Compute extra chat FNs when chat-eval was scored.
    if chat_link is not None:
        # Extra vs published TF-IDF FN count on the same 200 rows.
        extra_chat, extra_chat_url = _extra_fn_url_related(chat_link, tfidf_chat_fn)
    # Extra TEST URL-related FNs among LSTM FNs (cannot intersect TF-IDF FN ids).
    extra_test, extra_test_url = _extra_fn_url_related(test_link, tfidf_test_fn)

    # Chat-eval URL vs no-URL FN counts from the slice summary.
    chat_fn_url = int(chat_link["slices"]["fn_url"]) if chat_link is not None else 0
    # Chat-eval no-URL FN count.
    chat_fn_no_url = int(chat_link["slices"]["fn_no_url"]) if chat_link is not None else 0
    # TEST URL FN count.
    test_fn_url = int(test_link["slices"]["fn_url"])
    # TEST no-URL FN count.
    test_fn_no_url = int(test_link["slices"]["fn_no_url"])
    # LSTM URL-bearing scam recall on chat eval (None if no URL scams).
    lstm_url_chat = chat_link["slices"]["scam_recall_url"] if chat_link is not None else None
    # LSTM URL-bearing scam recall on TEST.
    lstm_url_test = test_link["slices"]["scam_recall_url"]

    # Apply A/B/C using published TF-IDF/DistilBERT overall metrics (no model reload).
    recommendation = recommend_char_lstm_exploration(
        chat_eval_fn_lstm=lstm_chat_fn,
        chat_eval_fn_tfidf=tfidf_chat_fn,
        test_fn_lstm=lstm_test_fn,
        test_fn_distilbert=distilbert_test_fn,
        test_scam_recall_lstm=lstm_test_recall,
        test_scam_recall_distilbert=distilbert_test_recall,
        test_scam_recall_tfidf=tfidf_test_recall,
        chat_eval_scam_recall_lstm=lstm_chat_recall,
        chat_eval_ham_warned_lstm=lstm_chat_fp,
        chat_eval_ham_warned_tfidf=tfidf_chat_fp,
        extra_fn_chat_eval_url_related=extra_chat_url,
        extra_fn_test_url_related=extra_test_url,
        lstm_url_scam_recall_chat=lstm_url_chat,
        lstm_url_scam_recall_test=lstm_url_test,
        tfidf_url_scam_recall_chat=None,
        tfidf_url_scam_recall_test=None,
        tfidf_scam_recall_chat=tfidf_chat_recall,
        chat_eval_fn_url=chat_fn_url,
        chat_eval_fn_no_url=chat_fn_no_url,
        test_fn_url=test_fn_url,
        test_fn_no_url=test_fn_no_url,
    )

    # Persist FN/FP URL-flag dumps plus slice summaries for TEST and chat eval.
    link_payload = {
        "note": (
            "TF-IDF and DistilBERT per-row error slices were not recomputed "
            "(checkpoints not loaded). Extra-FN counts subtract published FN "
            "totals. URL-bearing TF-IDF recall is unavailable; criterion C "
            "compares LSTM URL-bearing recall to TF-IDF overall scam recall."
        ),
        "test": _jsonify(test_link),
        "chat_eval": _jsonify(chat_link) if chat_link is not None else None,
        "extra_fn_chat_eval_vs_tfidf": extra_chat,
        "extra_fn_chat_eval_url_related": extra_chat_url,
        "extra_fn_test_vs_tfidf": extra_test,
        "extra_fn_test_url_related": extra_test_url,
        "char_lstm_recommendation": _jsonify(recommendation),
        "live_url_reputation": False,
    }
    # Write the link-error analysis next to the metrics JSON.
    link_path = args.reports_dir / "link_error_analysis.json"
    # Persist JSON for the char-LSTM decision and the README paragraph.
    link_path.write_text(json.dumps(link_payload, indent=2))
    # Point the operator at the link analysis.
    print(f"Wrote {link_path}")

    # Write the go/no-go markdown; do not implement a char LSTM in this pass.
    decision_path = args.reports_dir / "char_lstm_decision.md"
    # Render A/B/C from the recommendation payload.
    decision_path.write_text(render_char_lstm_decision_markdown(recommendation))
    # Point the operator at the decision file.
    print(f"Wrote {decision_path}  verdict={recommendation['verdict']}")


# Run the training pipeline only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
