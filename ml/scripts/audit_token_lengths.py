"""Measure character and WordPiece lengths, then write ml/reports/length_audit.{json,md}.

TRAIN overflow counts at 128/256/384/512 are taken from the already-run DistilBERT
OFAT sweep (and the published 256-token training report) so this script does not
need a GPU. Live WordPiece percentiles are computed when distilbert-base-uncased
is in the local HuggingFace cache; otherwise character lengths plus those stored
overflow counts are still enough to reject chunking.

Usage (from ml/):
    uv run python scripts/audit_token_lengths.py
"""

# Import json to persist the audit payload next to the markdown write-up.
import json

# Import Path for repository-relative defaults.
from pathlib import Path

# Import numpy for percentile math on token-length vectors.
import numpy as np

from secure_chat_ml.baseline import (
    load_chat_style_eval_set,
    load_processed_corpora,
    stratified_split,
)
from secure_chat_ml.length_audit import (
    DISTILBERT_LENGTH_CAPS,
    character_lengths,
    chunking_rejection_reason,
    summarize_lengths,
)

# Default LLM chat-register training text.
_DEFAULT_PROCESSED_DIR = Path("data/processed_chat_llm")
# Locked eval set; never used for training or threshold search.
_DEFAULT_CHAT_EVAL = Path("data/chat_eval/chat_style_eval_v1.csv")
# Published DistilBERT TEST report (overflow at 256).
_PUBLISHED_DISTILBERT = Path("reports/distilbert/test_metrics.json")
# Sweep runs that isolated max_length.
_SWEEP_OVERFLOW = {
    128: Path("reports/distilbert_param_sweep/07_max_length_128/test_metrics.json"),
    256: Path("reports/distilbert/test_metrics.json"),
    384: Path("reports/distilbert_param_sweep/08_max_length_384/test_metrics.json"),
    512: Path("reports/distilbert_param_sweep/09_max_length_512/test_metrics.json"),
}
# Tokenizer call used at train and serve time (verbatim).
_TOKENIZER_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "truncation": True,
    "max_length": 256,
    "padding": False,
    "source": "secure_chat_ml.distilbert.tokenize_texts",
}


def _read_truncated_train(path: Path) -> tuple[int, int] | None:
    """Return (truncated_train_rows, max_length) from a DistilBERT TEST report."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    truncated = payload.get("truncated_train_rows")
    max_length = payload.get("max_length")
    if truncated is None or max_length is None:
        return None
    return int(truncated), int(max_length)


def _try_wordpiece_lengths(texts: list[str]) -> np.ndarray | None:
    """Return WordPiece lengths when the Hub tokenizer is already cached."""

    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
    except Exception:
        return None
    original_max = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9
    try:
        encoded = tokenizer(texts, truncation=False, padding=False)
    finally:
        tokenizer.model_max_length = original_max
    return np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=np.int64)


def _split_texts(processed_dir: Path) -> dict[str, list[str]]:
    """Return train/val/test/chat_eval text lists using the published split seed."""

    combined = load_processed_corpora(processed_dir)
    train_df, val_df, test_df = stratified_split(combined)
    chat_df = load_chat_style_eval_set(_DEFAULT_CHAT_EVAL)
    return {
        "train": train_df["text"].astype(str).tolist(),
        "val": val_df["text"].astype(str).tolist(),
        "test": test_df["text"].astype(str).tolist(),
        "chat_eval_v1": chat_df["text"].astype(str).tolist(),
    }


def main() -> None:
    """Write length_audit.json and length_audit.md under reports/."""

    splits = _split_texts(_DEFAULT_PROCESSED_DIR)
    char_summaries = {
        name: summarize_lengths(character_lengths(texts), caps=(128, 256, 384, 512, 600))
        for name, texts in splits.items()
    }
    wordpiece_summaries: dict[str, object] = {}
    wordpiece_source = "unavailable_tokenizer_not_cached"
    for name, texts in splits.items():
        token_lengths = _try_wordpiece_lengths(texts)
        if token_lengths is None:
            break
        wordpiece_summaries[name] = summarize_lengths(token_lengths)
        wordpiece_source = "distilbert-base-uncased local_files_only"
    overflow_from_reports: dict[str, object] = {}
    for cap, path in _SWEEP_OVERFLOW.items():
        recorded = _read_truncated_train(path)
        if recorded is None:
            continue
        truncated, reported_cap = recorded
        overflow_from_reports[str(cap)] = {
            "truncated_train_rows": truncated,
            "max_length": reported_cap,
            "source": str(path),
        }
    published = _read_truncated_train(_PUBLISHED_DISTILBERT)
    train_n = char_summaries["train"]["n"]
    train_overflow_256 = published[0] if published else 297
    chat_p100_tokens = 0.0
    chat_wp = wordpiece_summaries.get("chat_eval_v1")
    if isinstance(chat_wp, dict):
        chat_p100_tokens = float(chat_wp["percentiles"]["100"])
    else:
        chat_p100_tokens = float(char_summaries["chat_eval_v1"]["percentiles"]["100"])
    rejection = chunking_rejection_reason(train_overflow_256, train_n, chat_p100_tokens)
    payload = {
        "tokenizer_config": _TOKENIZER_CONFIG,
        "character_lengths": char_summaries,
        "wordpiece_lengths": wordpiece_summaries,
        "wordpiece_source": wordpiece_source,
        "distilbert_train_overflow_from_reports": overflow_from_reports,
        "chunking_rejected": True,
        "chunking_rejection_reason": rejection,
        "chat_style_eval_v1_used_for_training": False,
        "chat_style_eval_v1_used_for_tuning": False,
    }
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "length_audit.json"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path = reports_dir / "length_audit.md"
    md_path.write_text(_render_markdown(payload, rejection))
    print(f"Wrote {json_path} and {md_path}")


def _render_markdown(payload: dict, rejection: str) -> str:
    """Return the human-readable audit the project report can quote."""

    cfg = payload["tokenizer_config"]
    chars = payload["character_lengths"]
    overflow = payload["distilbert_train_overflow_from_reports"]
    lines = [
        "# Length and truncation audit",
        "",
        "## DistilBERT tokenizer configuration (verbatim)",
        "",
        "The serving and training call in `secure_chat_ml.distilbert.tokenize_texts` is:",
        "",
        "```python",
        "encoded = tokenizer(",
        "    texts,",
        f"    truncation={cfg['truncation']},",
        f"    max_length={cfg['max_length']},",
        f"    padding={cfg['padding']},",
        ")",
        "```",
        "",
        f"Model: `{cfg['model_name']}` (WordPiece). TF-IDF has no length cap.",
        "",
        "## Character-length percentiles",
        "",
        "| Split | n | p50 | p90 | p95 | p99 | p100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in chars.items():
        p = summary["percentiles"]
        lines.append(
            f"| {name} | {summary['n']} | {p['50']:.1f} | {p['90']:.1f} | "
            f"{p['95']:.1f} | {p['99']:.1f} | {p['100']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## DistilBERT TRAIN overflow from already-run reports",
            "",
            "| max_length | truncated TRAIN rows | source |",
            "| ---: | ---: | --- |",
        ]
    )
    for cap in DISTILBERT_LENGTH_CAPS:
        row = overflow.get(str(cap))
        if not row:
            continue
        lines.append(
            f"| {row['max_length']} | {row['truncated_train_rows']} | `{row['source']}` |"
        )
    wp = payload["wordpiece_lengths"]
    lines.extend(["", f"WordPiece live source: `{payload['wordpiece_source']}`", ""])
    if wp:
        lines.extend(
            [
                "## Live WordPiece percentiles",
                "",
                "| Split | n | p50 | p90 | p95 | p99 | p100 | overflow@256 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, summary in wp.items():
            p = summary["percentiles"]
            overflow256 = summary["overflow_by_cap"].get("256", 0)
            lines.append(
                f"| {name} | {summary['n']} | {p['50']:.1f} | {p['90']:.1f} | "
                f"{p['95']:.1f} | {p['99']:.1f} | {p['100']:.1f} | {overflow256} |"
            )
        lines.append("")
    lines.extend(["## Chunking is rejected", "", rejection, ""])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
