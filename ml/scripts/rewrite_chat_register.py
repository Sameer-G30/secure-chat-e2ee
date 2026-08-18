"""Rewrite processed email/SMS corpora into WhatsApp/DM-style training text.

Reads `data/processed/*.csv` and writes `data/processed_chat/*.csv` with the
same schema columns plus `source_message_id`, `rewrite_method`, and
`urls_json`. Labels are copied unchanged. The locked chat-style eval set
under `data/chat_eval/` is never read or written.

Usage (from ml/):
    uv run python scripts/rewrite_chat_register.py
"""

# Import argparse so reviewers can point the rewriter at fixture directories.
import argparse

# Import json to persist urls_json cells and a row-count log.
import json

# Import Path for portable input/output locations.
from pathlib import Path

# Import pandas to load and write schema-shaped corpus CSVs.
import pandas as pd

# Import the documented rewrite identifier and the per-message rewriter.
from secure_chat_ml.chat_register import REWRITE_METHOD, rewrite_message

# Import URL extraction so urls_json records the links kept in each rewrite.
from secure_chat_ml.url_features import extract_urls

# Default to the same processed directory the download scripts already write.
_DEFAULT_INPUT_DIR = Path("data/processed")
# Write chat-register training text beside processed/, never into chat_eval/.
_DEFAULT_OUTPUT_DIR = Path("data/processed_chat")
# Persist a machine-readable in/dropped/urls_kept log next to other reports.
_DEFAULT_LOG_PATH = Path("reports/rewrite_chat_register_log.json")
# Canonical schema columns required by data/label-schema.yaml.
_SCHEMA_COLUMNS = [
    "message_id",
    "text",
    "label",
    "original_label",
    "source",
    "split",
]
# Extra columns that make the rewrite auditable without changing the schema.
_EXTRA_COLUMNS = [
    "source_message_id",
    "rewrite_method",
    "urls_json",
]


# Parse command-line arguments controlling input, output, and the count log.
def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments with repository-relative defaults."""

    # Build a parser from this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__)
    # Allow tests to point the rewriter at a tiny synthetic processed directory.
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help="Directory of normalized email/SMS CSVs (default: data/processed).",
    )
    # Keep the output directory distinct from data/chat_eval/ by construction.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for chat-register CSVs (default: data/processed_chat).",
    )
    # Persist in/dropped/urls_kept counts for the README and later audits.
    parser.add_argument(
        "--log-path",
        type=Path,
        default=_DEFAULT_LOG_PATH,
        help=(
            "JSON path for row-count statistics "
            "(default: reports/rewrite_chat_register_log.json)."
        ),
    )
    # Return the populated namespace for main().
    return parser.parse_args()


# Refuse to treat the locked eval directory as a rewrite source or destination.
def _assert_not_chat_eval(path: Path) -> None:
    """Raise ValueError if a path is inside the locked chat_eval directory."""

    # Inspect every part of the path so nested chat_eval/ copies are also rejected.
    if "chat_eval" in path.parts:
        # Fail loudly rather than silently mixing eval text into training.
        raise ValueError(
            f"Refusing to read or write {path}: the locked chat-style eval set "
            "must stay out of rewrite and training (chat_style_eval_training_allowed: false)."
        )


# Rewrite one processed corpus frame, dropping empty results and tracking stats.
def rewrite_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (rewritten_frame, stats) with labels copied unchanged."""

    # Count source rows before any drop so the log is comparable across runs.
    rows_in = len(frame)
    # Count rows dropped because the rewrite was empty after artifact stripping.
    dropped_empty = 0
    # Count rows whose original text contained at least one URL that was kept.
    urls_kept_rows = 0
    # Accumulate rewritten records for a single output DataFrame.
    records: list[dict[str, object]] = []
    # Iterate positional values to avoid depending on the Index.
    message_ids = frame["message_id"].astype(str).tolist()
    # Read source text, treating missing cells as empty so they can be dropped.
    texts = frame["text"].fillna("").astype(str).tolist()
    # Copy labels as integers; the rewriter must not flip them.
    labels = frame["label"].astype(int).tolist()
    # Preserve the human-readable original_label for auditability.
    original_labels = frame["original_label"].astype(str).tolist()
    # Preserve the corpus name so source_counts in later reports still work.
    sources = frame["source"].astype(str).tolist()
    # Preserve the schema split column (still unassigned until train_baseline).
    splits = frame["split"].astype(str).tolist()
    # Walk every row in lockstep.
    for message_id, text, label, original_label, source, split in zip(
        message_ids,
        texts,
        labels,
        original_labels,
        sources,
        splits,
        strict=True,
    ):
        # Harvest URLs from the original so the log and urls_json stay honest.
        original_urls = extract_urls(text)
        # Produce a chat-register line, or None when nothing usable remains.
        rewritten = rewrite_message(text, label)
        # Drop empty rewrites rather than emitting blank training rows.
        if rewritten is None:
            # Count this source row as dropped.
            dropped_empty += 1
            # Continue with the next source row.
            continue
        # Confirm original URLs still appear in the rewrite when any were found.
        kept_urls = [url for url in original_urls if url in rewritten]
        # Count the row if at least one original URL survived.
        if original_urls and kept_urls:
            # Increment the urls_kept row counter for the log.
            urls_kept_rows += 1
        # Append a schema-shaped record plus rewrite metadata.
        records.append(
            {
                "message_id": message_id,
                "text": rewritten,
                "label": int(label),
                "original_label": original_label,
                "source": source,
                "split": split,
                "source_message_id": message_id,
                "rewrite_method": REWRITE_METHOD,
                "urls_json": json.dumps(original_urls, ensure_ascii=True),
            }
        )
    # Build a DataFrame even when every row was dropped, so concat stays safe.
    rewritten_frame = pd.DataFrame.from_records(records, columns=_SCHEMA_COLUMNS + _EXTRA_COLUMNS)
    # Package the per-frame counters for the global log.
    stats = {
        "rows_in": rows_in,
        "dropped_empty": dropped_empty,
        "urls_kept_rows": urls_kept_rows,
        "rows_out_before_dedup": len(rewritten_frame),
    }
    # Return the rewritten frame and its counters.
    return rewritten_frame, stats


# Load, rewrite, globally dedupe on rewritten text, and write per-source CSVs.
def rewrite_corpora(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Rewrite every processed CSV and write deduplicated chat-register CSVs."""

    # Refuse to treat the locked eval directory as an input.
    _assert_not_chat_eval(input_dir)
    # Refuse to write rewritten training text into the locked eval directory.
    _assert_not_chat_eval(output_dir)
    # Find every normalized corpus CSV the download scripts have produced.
    csv_paths = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    # Fail loudly rather than writing an empty processed_chat directory.
    if not csv_paths:
        # Explain how to produce the missing inputs.
        raise FileNotFoundError(
            f"No processed corpus CSVs found under {input_dir}. "
            "Run the ml/scripts/download_*.py scripts first."
        )
    # Accumulate rewritten frames across corpora before a global dedupe.
    frames: list[pd.DataFrame] = []
    # Accumulate per-file counters for the log.
    per_source: dict[str, dict[str, int]] = {}
    # Track global in/dropped/urls_kept totals.
    rows_in = 0
    # Track empty rewrites dropped before dedupe.
    dropped_empty = 0
    # Track rows that kept at least one original URL.
    urls_kept_rows = 0
    # Rewrite each corpus file independently, then concatenate.
    for path in csv_paths:
        # Load one normalized corpus CSV.
        frame = pd.read_csv(path)
        # Rewrite every row, copying labels unchanged.
        print(f"Rewriting {path.name} ({len(frame)} rows)...")
        # Rewrite every row, copying labels unchanged.
        rewritten, stats = rewrite_frame(frame)
        print(
            f"  {path.name}: in={stats['rows_in']} "
            f"dropped_empty={stats['dropped_empty']} "
            f"urls_kept_rows={stats['urls_kept_rows']} "
            f"out={stats['rows_out_before_dedup']}"
        )
        # Add this corpus to the combined frame list.
        frames.append(rewritten)
        # Record per-file counters under the CSV stem (sms_spam, enron_spam, ...).
        per_source[path.stem] = stats
        # Accumulate global input rows.
        rows_in += stats["rows_in"]
        # Accumulate global empty-drop rows.
        dropped_empty += stats["dropped_empty"]
        # Accumulate global URL-kept rows.
        urls_kept_rows += stats["urls_kept_rows"]
    # Concatenate every corpus so cross-source duplicate text can be dropped once.
    combined = pd.concat(frames, ignore_index=True)
    # Remember the pre-dedupe size for the log.
    before_dedup = len(combined)
    # Drop exact-duplicate rewritten text the same way load_processed_corpora does.
    combined = combined.drop_duplicates(subset="text", keep="first")
    # Reset the index so per-source writes are positional-stable.
    combined = combined.reset_index(drop=True)
    # Count how many duplicate rewrites were removed.
    dedup_dropped = before_dedup - len(combined)
    # Create the output directory without failing when it already exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write one CSV per source so train_baseline can glob processed_chat/*.csv.
    for source_name, group in combined.groupby("source", sort=True):
        # Map source names onto the same filenames the download scripts used when possible.
        filename = {
            "uci_sms_spam": "sms_spam.csv",
            "enron_spam": "enron_spam.csv",
            "spamassassin": "spamassassin.csv",
            "nazario": "nazario.csv",
            "kaggle_phishing": "kaggle_phishing.csv",
        }.get(str(source_name), f"{source_name}.csv")
        # Write a portable UTF-8 CSV without a redundant DataFrame index.
        group.to_csv(output_dir / filename, index=False)
    # Package the global log payload.
    return {
        "rewrite_method": REWRITE_METHOD,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rows_in": rows_in,
        "dropped_empty": dropped_empty,
        "urls_kept_rows": urls_kept_rows,
        "rows_out_before_dedup": before_dedup,
        "dedup_dropped": dedup_dropped,
        "rows_out": len(combined),
        "per_source": per_source,
        "chat_eval_touched": False,
    }


# Run the full rewrite and persist the row-count log.
def main() -> None:
    """Rewrite processed corpora into data/processed_chat and print counts."""

    # Parse CLI arguments (defaults match the documented from-ml/ command).
    args = parse_args()
    # Rewrite every processed CSV into chat-register training text.
    log = rewrite_corpora(args.input_dir, args.output_dir)
    # Ensure the reports directory exists before writing the log.
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist the in/dropped/urls_kept log as JSON for audits.
    args.log_path.write_text(json.dumps(log, indent=2))
    # Print a concise human-readable summary without message content.
    print(
        f"in: {log['rows_in']}  dropped_empty: {log['dropped_empty']}  "
        f"urls_kept_rows: {log['urls_kept_rows']}  "
        f"dedup_dropped: {log['dedup_dropped']}  out: {log['rows_out']}"
    )
    # Point the operator at the output directory.
    print(f"Wrote chat-register CSVs under {args.output_dir} ({REWRITE_METHOD})")
    # Point the operator at the JSON log.
    print(f"Wrote {args.log_path}")


# Run the rewriter only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
