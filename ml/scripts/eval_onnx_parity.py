"""Compare an int8 ONNX graph with its fp32 sibling on TEST and the chat eval sets.

Closes the gap noted in the review: the shipped int8 DistilBERT was never scored on the full
TEST split, only on four browser fixtures. Reports precision/recall/F1 and ROC/PR-AUC for both
graphs and the deltas.

Usage (from ml/):
    uv run python scripts/eval_onnx_parity.py --export-dir exports/onnx_web/distilbert_default
"""

# Parse CLI options.
import argparse
import json
import time
from pathlib import Path

import numpy as np

from secure_chat_ml.baseline import (
    load_chat_style_eval_set,
    load_processed_corpora,
    stratified_split,
)
from secure_chat_ml.distilbert import load_saved_classifier
from secure_chat_ml.onnx_eval import mcnemar_p_value, metrics_at_threshold, score_onnx


def main() -> None:
    """Score fp32 and int8 graphs and write reports/onnx_parity/<name>.json."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir", type=Path, default=Path("exports/onnx_web/distilbert_default")
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_chat_llm"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--int8-name", type=str, default="model.onnx")
    parser.add_argument("--fp32-name", type=str, default="model.fp32.onnx")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N TEST rows.")
    args = parser.parse_args()

    # Same split/seed as training so TEST is genuinely held out.
    combined = load_processed_corpora(args.processed_dir)
    _train, _val, test_df = stratified_split(
        combined, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42
    )
    if args.limit:
        test_df = test_df.head(args.limit)
    sets = {
        "test": test_df,
        "chat_v1": load_chat_style_eval_set(Path("data/chat_eval/chat_style_eval_v1.csv")),
        "chat_v2": load_chat_style_eval_set(Path("data/chat_eval/chat_style_eval_v2.csv")),
    }
    _model, tokenizer = load_saved_classifier(args.model_dir)

    report: dict[str, object] = {
        "export_dir": str(args.export_dir),
        "threshold": args.threshold,
        "max_length": args.max_length,
        "int8_bytes": (args.export_dir / args.int8_name).stat().st_size,
        "fp32_bytes": (args.export_dir / args.fp32_name).stat().st_size,
    }
    for set_name, frame in sets.items():
        texts = frame["text"].astype(str).tolist()
        labels = frame["label"].to_numpy().astype(int)
        scores: dict[str, np.ndarray] = {}
        timings: dict[str, float] = {}
        for tag, filename in (("fp32", args.fp32_name), ("int8", args.int8_name)):
            started = time.perf_counter()
            scores[tag] = score_onnx(
                args.export_dir / filename, tokenizer, texts, max_length=args.max_length
            )
            timings[tag] = (time.perf_counter() - started) * 1000 / max(1, len(texts))
        m32 = metrics_at_threshold(labels, scores["fp32"], args.threshold)
        m8 = metrics_at_threshold(labels, scores["int8"], args.threshold)
        pred32 = (scores["fp32"] >= args.threshold).astype(int)
        pred8 = (scores["int8"] >= args.threshold).astype(int)
        report[set_name] = {
            "rows": len(texts),
            "fp32": m32,
            "int8": m8,
            "delta_int8_minus_fp32": {
                k: m8[k] - m32[k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")
            },
            "decision_flips": int((pred32 != pred8).sum()),
            "max_abs_score_diff": float(np.abs(scores["fp32"] - scores["int8"]).max()),
            "mcnemar_p": mcnemar_p_value(labels, pred32, pred8),
            "ms_per_message_cpu": timings,
        }
        print(set_name, json.dumps(report[set_name]["delta_int8_minus_fp32"], indent=None))  # type: ignore[index]

    out = Path("reports/onnx_parity") / f"{args.export_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
