"""Export a distilled student to ONNX (vocab-pruned, dynamic int8) as `distilbert_student`.

Usage (from ml/):
    uv run python scripts/export_student.py --name minilm_l6_s42 --export-name distilbert_student
"""

# Parse CLI options.
import argparse
import json
from pathlib import Path

from secure_chat_ml.baseline import load_processed_corpora, stratified_split
from secure_chat_ml.distill import load_student
from secure_chat_ml.onnx_export import export_distilbert_checkpoint
from secure_chat_ml.vocab_prune import count_token_usage, prune_embeddings, select_keep_ids


def main() -> None:
    """Prune, export, and copy the student into exports/onnx_web/<export-name>."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Directory name under models/students/.")
    parser.add_argument("--export-name", default="distilbert_student")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_chat_llm"))
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--frontend-public-ml", type=Path, default=None)
    args = parser.parse_args()

    summary = json.loads((Path("reports/students") / args.name / "summary.json").read_text())
    max_length = int(summary["max_length"])
    threshold = float(summary["threshold"])

    model, tokenizer = load_student(str(Path("models/students") / args.name))
    combined = load_processed_corpora(args.processed_dir)
    train_df, _val, _test = stratified_split(
        combined, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42
    )
    counts = count_token_usage(tokenizer, train_df["text"].astype(str).tolist(), max_length)
    keep = select_keep_ids(
        counts,
        min_count=args.min_count,
        special_ids=list(tokenizer.all_special_ids),
        vocab_size=len(tokenizer),
    )
    print(f"Keeping {len(keep)} of {len(tokenizer)} ids")
    prune_embeddings(model, keep, unk_id=tokenizer.unk_token_id)

    spec = {
        "id": args.export_name,
        "load_order": 99,
        "label": f"Distilled {summary['student']} (max_length {max_length}, thr {threshold})",
        "model_dir": str(Path("models/students") / args.name),
        "reports_dir": str(Path("reports/students") / args.name),
        "metrics_name": "test_metrics.json",
        "threshold": threshold,
        "max_length": max_length,
        "export_dirname": args.export_name,
    }
    out = export_distilbert_checkpoint(
        Path.cwd(),
        spec,
        Path("exports/onnx_web"),
        frontend_public_ml=args.frontend_public_ml,
        model_override=(model, tokenizer),
    )
    for name in ("model.fp32.onnx", "model.onnx"):
        print(name, round((out / name).stat().st_size / 2**20, 2), "MiB")


if __name__ == "__main__":
    main()
