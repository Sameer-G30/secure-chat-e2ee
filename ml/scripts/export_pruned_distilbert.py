"""Stage A of the compression plan: vocab-pruned (and optionally per-channel) DistilBERT export.

Same weights, same tokenizer, same input interface as distilbert_default; only the unused rows of
the 30522 x 768 embedding table are removed. No retraining.

Usage (from ml/):
    uv run python scripts/export_pruned_distilbert.py --min-count 2 --per-channel
"""

# Parse CLI options.
import argparse
from pathlib import Path

from secure_chat_ml.baseline import load_processed_corpora, stratified_split
from secure_chat_ml.distilbert import load_saved_classifier
from secure_chat_ml.onnx_export import export_distilbert_checkpoint
from secure_chat_ml.vocab_prune import count_token_usage, prune_embeddings, select_keep_ids


def main() -> None:
    """Prune the embedding table, export fp32 + int8 ONNX, and print the size result."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_chat_llm"))
    parser.add_argument("--reports-dir", type=str, default="reports/distilbert")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--name", type=str, default="distilbert_pruned")
    args = parser.parse_args()

    model, tokenizer = load_saved_classifier(args.model_dir)
    # Count usage on TRAIN only so TEST/chat remain genuinely unseen.
    combined = load_processed_corpora(args.processed_dir)
    train_df, _val, _test = stratified_split(
        combined, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42
    )
    counts = count_token_usage(tokenizer, train_df["text"].astype(str).tolist(), args.max_length)
    keep = select_keep_ids(
        counts,
        min_count=args.min_count,
        special_ids=list(tokenizer.all_special_ids),
        vocab_size=len(tokenizer),
    )
    print(f"Keeping {len(keep)} of {len(tokenizer)} WordPiece ids (min_count={args.min_count})")
    prune_embeddings(model, keep, unk_id=tokenizer.unk_token_id)

    spec = {
        "id": args.name,
        "load_order": 99,
        "label": f"DistilBERT pruned ({len(keep)} ids, per_channel={args.per_channel})",
        "model_dir": str(args.model_dir),
        "reports_dir": args.reports_dir,
        "metrics_name": "test_metrics.json",
        "threshold": args.threshold,
        "max_length": args.max_length,
        "export_dirname": args.name,
        "per_channel": args.per_channel,
    }
    out = export_distilbert_checkpoint(
        Path.cwd(), spec, Path("exports/onnx_web"), model_override=(model, tokenizer)
    )
    for name in ("model.fp32.onnx", "model.onnx"):
        print(name, round((out / name).stat().st_size / 2**20, 1), "MiB")


if __name__ == "__main__":
    main()
