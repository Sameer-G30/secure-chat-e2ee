"""Distill distilbert_best into a small student (MiniLM-L6-H384 or TinyBERT-4L).

Teacher logits are computed once on plain TRAIN text and cached. Half of the student's inputs are
wrapped in synthetic `Me:/Them:` multi-turn windows (train/serve skew fix). The decision threshold
is frozen on VAL (same mixed plain/windowed composition) with the repo's existing selection rule.
TEST and the locked chat-eval v1 are scored predict-only.

Usage (from ml/):
    uv run python scripts/train_distilled_student.py --student nreimers/MiniLM-L6-H384-uncased \\
        --name minilm_l6_s42 --seed 42
"""

# Parse CLI options.
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from secure_chat_ml.baseline import (
    load_chat_style_eval_set,
    load_processed_corpora,
    stratified_split,
)
from secure_chat_ml.distilbert import (
    DISTILBERT_EXPANDED_THRESHOLD_GRID,
    evaluate_from_proba,
    load_saved_classifier,
    predict_scam_proba,
    tune_threshold_on_validation,
)
from secure_chat_ml.distill import distill_student, load_student, teacher_logits
from secure_chat_ml.multiturn import build_multiturn_frame

# Keep HF tokenizers from spawning extra threads under WSL2.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")


def jsonable(value: object) -> object:
    """Convert numpy scalars in nested sklearn reports into plain JSON types."""

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def benign_chat_pool(train_df: pd.DataFrame) -> list[str]:
    """Short legitimate TRAIN messages used as conversational context."""

    legit = train_df[train_df["label"] == 0]["text"].astype(str)
    return legit[legit.str.len() < 200].tolist()


def main() -> None:
    """Train one student seed and write reports/students/<name>/."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="nreimers/MiniLM-L6-H384-uncased")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-dir", type=Path, default=Path("models/distilbert_param_sweep/09_max_length_512")
    )
    parser.add_argument("--teacher-max-length", type=int, default=512)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_chat_llm"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--window-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: use N train rows.")
    args = parser.parse_args()

    # Same split and seed as the baseline so TEST is identical.
    combined = load_processed_corpora(args.processed_dir)
    train_df, val_df, test_df = stratified_split(
        combined, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42
    )
    if args.limit:
        train_df = train_df.head(args.limit)
    train_df = train_df.reset_index(drop=True)

    # Teacher soft targets on the plain current message (cached across seeds).
    cache = Path("models/students/_teacher_cache") / f"{args.teacher_dir.name}_{len(train_df)}.npy"
    if cache.exists():
        soft = np.load(cache)
    else:
        teacher, teacher_tok = load_saved_classifier(args.teacher_dir)
        soft = teacher_logits(
            teacher,
            teacher_tok,
            train_df["text"].astype(str).tolist(),
            max_length=args.teacher_max_length,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, soft)
        del teacher

    # Window half of TRAIN and VAL with benign context; targets still come from the plain text.
    pool = benign_chat_pool(train_df)
    train_in = build_multiturn_frame(train_df, pool, seed=args.seed, fraction=args.window_fraction)
    val_in = build_multiturn_frame(val_df, pool, seed=7, fraction=args.window_fraction)

    student, tokenizer = load_student(args.student)
    model_dir = Path("models/students") / args.name
    distill_student(
        student,
        tokenizer,
        train_in["text"].tolist(),
        train_in["label"].astype(int).tolist(),
        soft,
        output_dir=model_dir / "_scratch",
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        temperature=args.temperature,
        seed=args.seed,
    )
    student.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Freeze the threshold on VAL only.
    val_proba = predict_scam_proba(student, tokenizer, val_in["text"], max_length=args.max_length)
    tuning = tune_threshold_on_validation(
        val_in["label"], val_proba, threshold_grid=DISTILBERT_EXPANDED_THRESHOLD_GRID
    )
    report_dir = Path("reports/students") / args.name
    report_dir.mkdir(parents=True, exist_ok=True)

    sets = {
        "test_metrics.json": test_df,
        "chat_style_eval_metrics.json": load_chat_style_eval_set(
            Path("data/chat_eval/chat_style_eval_v1.csv")
        ),
    }
    summary: dict[str, object] = {
        "student": args.student,
        "seed": args.seed,
        "threshold": tuning.threshold,
        "max_length": args.max_length,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }
    for filename, frame in sets.items():
        proba = predict_scam_proba(student, tokenizer, frame["text"], max_length=args.max_length)
        result = evaluate_from_proba(
            frame["label"],
            proba,
            threshold=tuning.threshold,
            train_rows=len(train_df),
            source_frame=frame,
            val_rows=len(val_df),
        )
        payload = {
            "chosen_threshold": tuning.threshold,
            "classification_report": jsonable(result.classification_report),
            "confusion_matrix": result.confusion_matrix,
            **summary,
        }
        (report_dir / filename).write_text(json.dumps(payload, indent=2))
        summary[filename.removesuffix(".json")] = {
            "accuracy": result.classification_report["accuracy"],
            "scam_f1": result.classification_report["1"]["f1-score"]
            if "1" in result.classification_report
            else result.classification_report["scam"]["f1-score"],
        }
    (report_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    print(json.dumps(jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
