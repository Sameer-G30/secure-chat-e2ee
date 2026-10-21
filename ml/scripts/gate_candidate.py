"""Acceptance gate: does a smaller int8 ONNX candidate match the shipped distilbert_default?

Scores both int8 graphs on the held-out TEST split plus chat eval v1 (locked, predict-only) and v2,
then checks the gate defined in the compression plan. Everything is measured on the int8 ONNX
graphs that the browser would actually run, not on PyTorch weights.

Usage (from ml/):
    uv run python scripts/gate_candidate.py --candidate-dir exports/onnx_web/distilbert_pruned_pc \\
        --candidate-tokenizer models/distilbert --candidate-threshold 0.30 --max-mib 60
"""

# Parse CLI options.
import argparse
import json
from pathlib import Path

import numpy as np

from secure_chat_ml.baseline import (
    load_chat_style_eval_set,
    load_processed_corpora,
    stratified_split,
)
from secure_chat_ml.distilbert import load_saved_classifier
from secure_chat_ml.onnx_eval import mcnemar_p_value, metrics_at_threshold, score_onnx

# Tolerances versus the shipped baseline (absolute, on the scam class / AUC).
F1_TOLERANCE = 0.003
RECALL_TOLERANCE = 0.003
AUC_TOLERANCE = 0.002
# Locked chat-v1 accuracy of the shipped default (reports/distilbert/chat_style_eval_metrics).
CHAT_V1_MIN_ACCURACY = 0.895


def cached_scores(path: Path, onnx: Path, tokenizer, texts, max_length) -> np.ndarray:
    """Score texts, caching the result so repeated gate runs reuse the baseline."""

    if path.exists():
        cached = np.load(path)
        if len(cached) == len(texts):
            return cached
    scores = score_onnx(onnx, tokenizer, texts, max_length=max_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, scores)
    return scores


def main() -> None:
    """Run the gate and write reports/compression/<candidate>.json."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-tokenizer", type=Path, default=Path("models/distilbert"))
    parser.add_argument("--candidate-threshold", type=float, required=True)
    parser.add_argument("--candidate-max-length", type=int, default=256)
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("exports/onnx_web/distilbert_default")
    )
    parser.add_argument("--baseline-tokenizer", type=Path, default=Path("models/distilbert"))
    parser.add_argument("--baseline-threshold", type=float, default=0.30)
    parser.add_argument("--baseline-max-length", type=int, default=256)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_chat_llm"))
    parser.add_argument("--max-mib", type=float, default=25.0)
    args = parser.parse_args()

    combined = load_processed_corpora(args.processed_dir)
    _train, _val, test_df = stratified_split(
        combined, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42
    )
    sets = {
        "test": test_df,
        "chat_v1": load_chat_style_eval_set(Path("data/chat_eval/chat_style_eval_v1.csv")),
        "chat_v2": load_chat_style_eval_set(Path("data/chat_eval/chat_style_eval_v2.csv")),
    }
    _m1, base_tok = load_saved_classifier(args.baseline_tokenizer)
    _m2, cand_tok = load_saved_classifier(args.candidate_tokenizer)
    cache = Path("reports/compression/_cache")

    report: dict[str, object] = {"candidate": str(args.candidate_dir)}
    cand_bytes = (args.candidate_dir / "model.onnx").stat().st_size
    report["candidate_mib"] = cand_bytes / 2**20
    report["baseline_mib"] = (args.baseline_dir / "model.onnx").stat().st_size / 2**20
    verdicts: dict[str, bool] = {}
    pooled_true, pooled_base, pooled_cand = [], [], []
    for name, frame in sets.items():
        texts = frame["text"].astype(str).tolist()
        labels = frame["label"].to_numpy().astype(int)
        base_scores = cached_scores(
            cache / f"baseline_{name}.npy",
            args.baseline_dir / "model.onnx",
            base_tok,
            texts,
            args.baseline_max_length,
        )
        cand_scores = score_onnx(
            args.candidate_dir / "model.onnx",
            cand_tok,
            texts,
            max_length=args.candidate_max_length,
        )
        base_m = metrics_at_threshold(labels, base_scores, args.baseline_threshold)
        cand_m = metrics_at_threshold(labels, cand_scores, args.candidate_threshold)
        base_pred = (base_scores >= args.baseline_threshold).astype(int)
        cand_pred = (cand_scores >= args.candidate_threshold).astype(int)
        report[name] = {
            "rows": len(texts),
            "baseline": base_m,
            "candidate": cand_m,
            "mcnemar_p_vs_baseline": mcnemar_p_value(labels, base_pred, cand_pred),
        }
        if name != "test":
            pooled_true.append(labels)
            pooled_base.append(base_pred)
            pooled_cand.append(cand_pred)
        print(f"{name} baseline f1 {base_m['f1']:.4f} | candidate f1 {cand_m['f1']:.4f}")
        if name == "test":
            verdicts["test_f1"] = cand_m["f1"] >= base_m["f1"] - F1_TOLERANCE
            verdicts["test_recall"] = cand_m["recall"] >= base_m["recall"] - RECALL_TOLERANCE
            verdicts["test_roc_auc"] = cand_m["roc_auc"] >= base_m["roc_auc"] - AUC_TOLERANCE
        if name == "chat_v1":
            verdicts["chat_v1_accuracy"] = cand_m["accuracy"] >= CHAT_V1_MIN_ACCURACY
    # Pool both chat sets (400 rows) so McNemar has some power; n=200 alone is too noisy.
    p_pooled = mcnemar_p_value(
        np.concatenate(pooled_true), np.concatenate(pooled_base), np.concatenate(pooled_cand)
    )
    report["chat_pooled_mcnemar_p"] = p_pooled
    verdicts["chat_no_significant_regression"] = p_pooled > 0.05 or float(
        np.mean(np.concatenate(pooled_cand) == np.concatenate(pooled_true))
    ) >= float(np.mean(np.concatenate(pooled_base) == np.concatenate(pooled_true)))
    verdicts["size"] = report["candidate_mib"] <= args.max_mib  # type: ignore[operator]
    report["verdicts"] = verdicts
    report["passed"] = all(verdicts.values())
    out = Path("reports/compression") / f"{args.candidate_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(verdicts, indent=2), "PASSED" if report["passed"] else "FAILED")


if __name__ == "__main__":
    main()
