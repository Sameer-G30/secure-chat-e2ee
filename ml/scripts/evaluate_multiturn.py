"""Score a DistilBERT checkpoint on chat eval v2 as plain messages AND as multi-turn windows.

This exposes the train/serve skew: the browser scores `Me:/Them:` windows but the models
were trained on single messages. v2 is the development set; the locked v1 is never used.

Usage (from ml/):
    uv run python scripts/evaluate_multiturn.py --model-dir models/distilbert --threshold 0.30
"""

# Parse CLI options.
import argparse

# Write the JSON report.
import json
from pathlib import Path

from secure_chat_ml.baseline import load_chat_style_eval_set
from secure_chat_ml.distilbert import evaluate_from_proba, load_saved_classifier, predict_scam_proba
from secure_chat_ml.multiturn import build_multiturn_frame


def main() -> None:
    """Evaluate plain vs windowed text for one checkpoint and write a report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert"))
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--eval-path", type=Path, default=Path("data/chat_eval/chat_style_eval_v2.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/multiturn_eval.json"))
    args = parser.parse_args()

    # Refuse the locked v1 file; it must stay predict-only for the published numbers.
    if "v1" in args.eval_path.name:
        raise SystemExit("Use the v2 development set, not the locked v1 file.")

    frame = load_chat_style_eval_set(args.eval_path)
    # Benign priors only, so context never leaks the label.
    pool = frame.loc[frame["label"] == 0, "text"].astype(str).tolist()
    windowed = build_multiturn_frame(frame, pool, seed=42)

    model, tokenizer = load_saved_classifier(args.model_dir)
    report: dict[str, object] = {"model_dir": str(args.model_dir), "threshold": args.threshold}
    for name, data in (("plain", frame), ("multiturn", windowed)):
        proba = predict_scam_proba(model, tokenizer, data["text"], max_length=args.max_length)
        result = evaluate_from_proba(
            data["label"], proba, threshold=args.threshold, train_rows=0, source_frame=data
        )
        report[name] = {
            "accuracy": float(result.classification_report["accuracy"]),
            "confusion_matrix": result.confusion_matrix,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
