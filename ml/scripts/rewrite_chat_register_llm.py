"""Rewrite processed email/SMS corpora into intent-preserving DM-style text.

Reads `data/processed/*.csv` and writes `data/processed_chat_llm/*.csv` with
the same schema columns plus `source_message_id`, `rewrite_method`, and
`urls_json`. Labels are copied unchanged. The locked chat-style eval set
under `data/chat_eval/` is never read or written.

A local Ollama model (default: llama3.2:latest, already pulled on this
machine) paraphrases each source into a short WhatsApp/iMessage/DM. This is
not header-stripping plus truncation. Python post-conditions re-attach any
original URL the model omitted. Assistant refusals are retried once with a
research/register-only prompt, then fall back to rule_based_v1 for that row
so labeled scams are not dropped. Corpus text is sent only to localhost Ollama.

Checkpointing: rows are stored in `data/processed_chat_llm/_rewrite_checkpoint.sqlite`
as they complete so a crash can `--resume` without redoing finished rows.
Do not reuse a checkpoint from a run that stored refusals as status=ok;
start with `--no-resume` (this deletes the sidecar and old CSVs).

Runtime (WSL2, llama3.2:latest, RTX 4060 8GB): ~0.4–0.8s/row after warmup,
plus a second generate on refusals. Full processed corpora (~71k rows) ≈
8–12 hours. Use `--resume` after a crash. `--limit N` is a smoke run.
`--stratified-sample 10000` is a label+source-stratified subset.

Usage (from ml/):
    uv run python scripts/rewrite_chat_register_llm.py --no-resume
    uv run python scripts/rewrite_chat_register_llm.py --resume
    uv run python scripts/rewrite_chat_register_llm.py --limit 20
    uv run python scripts/rewrite_chat_register_llm.py --stratified-sample 10000
    uv run python scripts/rewrite_chat_register_llm.py --resume --model llama3.2:latest
"""

# Import argparse so reviewers can point the rewriter at fixture directories.
import argparse

# Import json to persist urls_json cells and a row-count log.
import json

# Import sqlite3 for crash-safe incremental checkpoints.
import sqlite3

# Import time to record wall-clock runtime in the JSON log.
import time

# Import Path for portable input/output locations.
from pathlib import Path

# Import pandas to load and write schema-shaped corpus CSVs.
import pandas as pd

# Import train_test_split for label+source stratified subsampling.
from sklearn.model_selection import train_test_split

# Import the documented rewrite identifier and the per-message LLM rewriter.
from secure_chat_ml.chat_register_llm import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    FALLBACK_REWRITE_METHOD,
    LLM_MAX_REWRITE_CHARS,
    OLLAMA_NUM_PREDICT,
    REWRITE_METHOD,
    GenerateFn,
    assert_not_chat_eval_path,
    build_ollama_generate,
    is_unusable_llm_output,
    rewrite_message_llm,
)

# Default to the same processed directory the download scripts already write.
_DEFAULT_INPUT_DIR = Path("data/processed")
# Write LLM chat-register training text beside processed_chat, never into chat_eval/.
_DEFAULT_OUTPUT_DIR = Path("data/processed_chat_llm")
# Persist a machine-readable in/dropped/urls_kept/failed log next to other reports.
_DEFAULT_LOG_PATH = Path("reports/rewrite_chat_register_llm_log.json")
# SQLite sidecar filename stored inside the output directory.
_CHECKPOINT_NAME = "_rewrite_checkpoint.sqlite"
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
# Map corpus `source` values onto the same filenames the download scripts used.
_SOURCE_FILENAMES = {
    "uci_sms_spam": "sms_spam.csv",
    "enron_spam": "enron_spam.csv",
    "spamassassin": "spamassassin.csv",
    "nazario": "nazario.csv",
    "kaggle_phishing": "kaggle_phishing.csv",
}


# Parse command-line arguments controlling input, output, sampling, and Ollama.
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
        help="Directory for LLM chat-register CSVs (default: data/processed_chat_llm).",
    )
    # Persist in/dropped/urls_kept/llm_failed counts for the README and later audits.
    parser.add_argument(
        "--log-path",
        type=Path,
        default=_DEFAULT_LOG_PATH,
        help=(
            "JSON path for row-count statistics "
            "(default: reports/rewrite_chat_register_llm_log.json)."
        ),
    )
    # Optional SQLite checkpoint path; default is inside the output directory.
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="SQLite checkpoint path (default: <output-dir>/_rewrite_checkpoint.sqlite).",
    )
    # Smoke-run cap; applied after optional stratified sampling.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Rewrite at most N rows (smoke run). Mutually exclusive with "
        "--stratified-sample.",
    )
    # Faster training iteration: stratified subset of the full processed corpora.
    parser.add_argument(
        "--stratified-sample",
        type=int,
        default=None,
        help="Rewrite N rows stratified on label+source (e.g. 10000). "
        "Default still attempts the full processed corpora. Mutually exclusive "
        "with --limit.",
    )
    # Seed for stratified sampling so a 10k run is reproducible.
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for --stratified-sample (default: 42).",
    )
    # Resume from the SQLite sidecar so crashes do not redo finished rows.
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already marked ok in the checkpoint (default: on).",
    )
    # Local Ollama model name; must already be pulled (this script will not pull).
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Ollama model name (default: llama3.2:latest if pulled, else the "
            "first suitable local instruct model)."
        ),
    )
    # Local Ollama origin; non-loopback hosts are refused in chat_register_llm.py.
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=DEFAULT_OLLAMA_HOST,
        help="Local Ollama origin (default: http://127.0.0.1:11434).",
    )
    # Per-call HTTP timeout before a generate attempt is treated as a failure.
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_OLLAMA_TIMEOUT,
        help="Seconds to wait for one Ollama generate call (default: 90).",
    )
    # How often to print running in/dropped/urls_kept/failed counters.
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N processed source rows (default: 25).",
    )
    # Return the populated namespace for main().
    return parser.parse_args()


# Open (or create) the SQLite checkpoint used for crash resume.
def _checkpoint_connect(path: Path) -> sqlite3.Connection:
    """Return a connection with the rewrites table ready for INSERT OR REPLACE."""

    # Create parent directories so a fresh output dir can hold the sidecar.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the SQLite file, creating it when this is the first run.
    conn = sqlite3.connect(path)
    # Use WAL so a crash is less likely to corrupt the checkpoint.
    conn.execute("PRAGMA journal_mode=WAL")
    # Create the checkpoint table if this is a new file.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rewrites (
            source TEXT NOT NULL,
            message_id TEXT NOT NULL,
            text TEXT,
            label INTEGER NOT NULL,
            original_label TEXT,
            split TEXT,
            urls_json TEXT,
            status TEXT NOT NULL,
            urls_appended INTEGER NOT NULL DEFAULT 0,
            rewrite_method TEXT,
            PRIMARY KEY (source, message_id)
        )
        """
    )
    # Add rewrite_method on older sidecars so a mid-run schema bump can resume.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(rewrites)").fetchall()}
    # Only ALTER when this file was created before rewrite_method existed.
    if "rewrite_method" not in columns:
        # Store llm_intent_v1 vs rule_based_v1_fallback per row.
        conn.execute("ALTER TABLE rewrites ADD COLUMN rewrite_method TEXT")
    # Persist the schema immediately.
    conn.commit()
    # Return the open connection for the rewrite loop.
    return conn


# Load (source, message_id) pairs that should be skipped on --resume.
def _completed_keys(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Return keys whose checkpoint status is ok or empty (do not retry empty)."""

    # Query only terminal statuses; llm_failed rows are retried on resume.
    rows = conn.execute(
        "SELECT source, message_id FROM rewrites WHERE status IN ('ok', 'empty')"
    ).fetchall()
    # Build a set for O(1) membership tests in the rewrite loop.
    return {(str(source), str(message_id)) for source, message_id in rows}


# Persist one rewrite attempt (success or failure) into the checkpoint.
def _checkpoint_row(
    conn: sqlite3.Connection,
    *,
    source: str,
    message_id: str,
    text: str | None,
    label: int,
    original_label: str,
    split: str,
    urls_json: str,
    status: str,
    urls_appended: bool,
    rewrite_method: str | None = None,
) -> None:
    """INSERT OR REPLACE one checkpoint row and commit immediately."""

    # Replace any previous llm_failed attempt for this key.
    conn.execute(
        """
        INSERT OR REPLACE INTO rewrites (
            source, message_id, text, label, original_label, split,
            urls_json, status, urls_appended, rewrite_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            message_id,
            text,
            int(label),
            original_label,
            split,
            urls_json,
            status,
            int(urls_appended),
            rewrite_method,
        ),
    )
    # Commit per row so a crash keeps all finished work.
    conn.commit()


# Take a reproducible label+source stratified subset for faster runs.
def take_stratified_sample(
    frame: pd.DataFrame,
    n: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return n rows stratified on label+source, or the full frame if n is larger."""

    # Reject non-positive sample sizes rather than silently returning nothing.
    if n <= 0:
        # Fail with a message that names the flag.
        raise ValueError("--stratified-sample must be a positive integer")
    # If the operator asked for at least the full corpus, skip sampling.
    if n >= len(frame):
        # Return a positional-stable copy.
        return frame.reset_index(drop=True)
    # Build a stratum key so both class and corpus identity are preserved.
    strata = frame["label"].astype(str) + "|" + frame["source"].astype(str)
    # Prefer the two-way stratum; fall back to label-only if a rare cell breaks sklearn.
    try:
        # Hold out the remainder so exactly n rows are returned.
        sampled, _rest = train_test_split(
            frame,
            train_size=n,
            random_state=random_state,
            stratify=strata,
        )
    except ValueError:
        # A singleton stratum can make label+source stratification fail.
        sampled, _rest = train_test_split(
            frame,
            train_size=n,
            random_state=random_state,
            stratify=frame["label"],
        )
    # Return a clean 0-based index for the rewrite loop.
    return sampled.reset_index(drop=True)


# Map a corpus source name onto the filename train_baseline.py will glob.
def _filename_for_source(source_name: str) -> str:
    """Return the canonical CSV filename for a processed corpus source."""

    # Use the download-script names when the source is one of the five corpora.
    return _SOURCE_FILENAMES.get(str(source_name), f"{source_name}.csv")


# Write one CSV per source from successful checkpoint rows, after text dedupe.
def write_output_csvs(conn: sqlite3.Connection, output_dir: Path) -> tuple[int, int]:
    """Return (rows_out_before_dedup, rows_out) after writing per-source CSVs."""

    # Load every successful rewrite from the checkpoint.
    successful = pd.read_sql_query(
        """
        SELECT message_id, text, label, original_label, source, split,
               urls_json, rewrite_method
        FROM rewrites
        WHERE status = 'ok' AND text IS NOT NULL AND TRIM(text) != ''
        """,
        conn,
    )
    # Nothing to write when every row failed or the run has not started.
    if successful.empty:
        # Still create the output directory so the operator sees a known location.
        output_dir.mkdir(parents=True, exist_ok=True)
        # Report zeros so the log stays well-formed.
        return 0, 0
    # Copy source_message_id from the original id; labels were never flipped.
    successful["source_message_id"] = successful["message_id"]
    # Fill a missing method on older checkpoint rows; do not overwrite fallbacks.
    successful["rewrite_method"] = successful["rewrite_method"].fillna(REWRITE_METHOD)
    # Reorder columns to match the documented schema plus extras.
    successful = successful[_SCHEMA_COLUMNS + _EXTRA_COLUMNS]
    # Never write a stored assistant refusal; fallback rows should not match this.
    usable_mask = ~successful["text"].astype(str).map(is_unusable_llm_output)
    # Keep only rows that are actual DMs, not "I cannot write a scam" refusals.
    successful = successful.loc[usable_mask]
    # Remember the pre-dedupe size for the log.
    before_dedup = len(successful)
    # Drop exact-duplicate rewritten text the same way load_processed_corpora does.
    successful = successful.drop_duplicates(subset="text", keep="first")
    # Reset the index so per-source writes are positional-stable.
    successful = successful.reset_index(drop=True)
    # Create the output directory without failing when it already exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write one CSV per source so train_baseline can glob processed_chat_llm/*.csv.
    for source_name, group in successful.groupby("source", sort=True):
        # Map source names onto the same filenames the download scripts used when possible.
        filename = _filename_for_source(str(source_name))
        # Write a portable UTF-8 CSV without a redundant DataFrame index.
        group.to_csv(output_dir / filename, index=False)
    # Return pre- and post-dedupe counts for the JSON log.
    return before_dedup, len(successful)


# Load, optionally sample, rewrite with resume, globally dedupe, and write CSVs.
def rewrite_corpora(
    input_dir: Path,
    output_dir: Path,
    *,
    generate: GenerateFn | None = None,
    model: str | None = None,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    resume: bool = True,
    limit: int | None = None,
    stratified_sample: int | None = None,
    random_state: int = 42,
    progress_every: int = 25,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Rewrite processed CSVs into intent-preserving DMs with a resume checkpoint."""

    # Refuse to treat the locked eval directory as an input.
    assert_not_chat_eval_path(input_dir)
    # Refuse to write rewritten training text into the locked eval directory.
    assert_not_chat_eval_path(output_dir)
    # --limit and --stratified-sample mean different things; do not combine them.
    if limit is not None and stratified_sample is not None:
        # Fail rather than silently applying both.
        raise ValueError("Use only one of --limit and --stratified-sample")
    # Find every normalized corpus CSV the download scripts have produced.
    csv_paths = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    # Fail loudly rather than writing an empty processed_chat_llm directory.
    if not csv_paths:
        # Explain how to produce the missing inputs.
        raise FileNotFoundError(
            f"No processed corpus CSVs found under {input_dir}. "
            "Run the ml/scripts/download_*.py scripts first."
        )
    # Resolve the checkpoint path inside the output dir unless the caller overrode it.
    checkpoint = checkpoint_path or (output_dir / _CHECKPOINT_NAME)
    # Refuse a checkpoint that would live under chat_eval/.
    assert_not_chat_eval_path(checkpoint)
    # A fresh run must not mix leftover checkpoint rows into the output CSVs.
    if not resume:
        # Drop the previous SQLite sidecar when the operator asked for a clean start.
        if checkpoint.exists():
            # Remove the checkpoint file so INSERT starts from an empty table.
            checkpoint.unlink()
        # Also drop WAL/SHM sidecars if a previous crash left them behind.
        for suffix in ("-wal", "-shm"):
            # Build the sidecar path next to the checkpoint file.
            extra = Path(str(checkpoint) + suffix)
            # Remove the extra file when it exists.
            if extra.exists():
                # Unlink the WAL/SHM leftover.
                extra.unlink()
        # Drop previously written CSVs so a smaller rerun cannot keep stale sources.
        if output_dir.exists():
            # Only unlink CSV products, never chat_eval files (path already checked).
            for old_csv in output_dir.glob("*.csv"):
                # Remove the stale chat-register CSV.
                old_csv.unlink()
    # Open (or create) the SQLite sidecar used for crash resume.
    conn = _checkpoint_connect(checkpoint)
    # Resolve the generate callback: tests inject a fake; the CLI uses local Ollama.
    resolved_model = "injected_generate"
    # Only contact Ollama when the caller did not inject a generate function.
    if generate is None:
        # Build a localhost-only callback and record the exact pulled model name.
        generate, resolved_model = build_ollama_generate(
            model=model or DEFAULT_OLLAMA_MODEL,
            host=ollama_host,
            timeout=timeout,
        )
    # Tests inject a callback; the CLI always has a generator after the block above.
    assert generate is not None
    # Accumulate source frames before optional sampling.
    frames: list[pd.DataFrame] = []
    # Load each corpus file independently, then concatenate.
    for path in csv_paths:
        # Load one normalized corpus CSV.
        frame = pd.read_csv(path)
        # Require the schema columns so a mis-pointed directory fails early.
        missing = [column for column in _SCHEMA_COLUMNS if column not in frame.columns]
        # Fail rather than inventing labels or ids.
        if missing:
            # Name the file and the missing columns.
            raise ValueError(f"{path} is missing required columns: {missing}")
        # Keep this corpus for the combined frame.
        frames.append(frame)
    # Concatenate every corpus so stratified sampling can see all sources.
    combined = pd.concat(frames, ignore_index=True)
    # Record how many processed rows existed before sampling/limit.
    rows_available = len(combined)
    # Track how the operator subset the corpus for the log.
    sample_mode = "full"
    # Apply label+source stratified sampling when requested.
    if stratified_sample is not None:
        # Subset to N rows while preserving class and source mix.
        combined = take_stratified_sample(combined, stratified_sample, random_state)
        # Record that this run is a stratified subset, not the full 71k.
        sample_mode = "stratified"
    # Apply a smoke-run cap after any sampling (mutually exclusive in practice).
    if limit is not None:
        # Reject a non-positive limit rather than rewriting zero rows silently.
        if limit <= 0:
            # Name the flag in the error.
            raise ValueError("--limit must be a positive integer")
        # Sort so a --limit smoke run is deterministic across machines.
        combined = combined.sort_values(["source", "message_id"], kind="mergesort")
        # Keep only the first N rows of the stable order.
        combined = combined.head(limit).reset_index(drop=True)
        # Record that this run was a smoke limit (unless already marked stratified).
        sample_mode = "limit"
    # Count source rows in the selected subset before any drop.
    rows_in = len(combined)
    # Skip keys already completed when --resume is on.
    done_keys = _completed_keys(conn) if resume else set()
    # Count rows skipped because they were already ok/empty in the checkpoint.
    resumed_skipped = 0
    # Count rows dropped because the source text was empty.
    dropped_empty = 0
    # Count rows where the LLM produced a real DM (not a refusal).
    llm_ok = 0
    # Count rows kept via rule_based_v1 after a refusal/garbage LLM output.
    llm_refused_then_fallback = 0
    # Count rows skipped after retry+fallback still produced nothing usable.
    llm_failed = 0
    # Count rows whose original text contained URLs that remain in the rewrite.
    urls_kept_rows = 0
    # Count rows where Python appended at least one URL the model omitted.
    urls_appended_rows = 0
    # Count rows newly written as ok in this process (excludes resume skips).
    newly_ok = 0
    # Print the model and subset size before the long loop.
    print(
        f"Rewriting {rows_in} rows with {REWRITE_METHOD} "
        f"(fallback={FALLBACK_REWRITE_METHOD}, model={resolved_model}, "
        f"sample_mode={sample_mode}, resume={resume}, "
        f"num_predict={OLLAMA_NUM_PREDICT}, llm_max_chars={LLM_MAX_REWRITE_CHARS})"
    )
    # Iterate positional values to avoid depending on the Index.
    message_ids = combined["message_id"].astype(str).tolist()
    # Read source text, treating missing cells as empty so they can be dropped.
    texts = combined["text"].fillna("").astype(str).tolist()
    # Copy labels as integers; the rewriter must not flip them.
    labels = combined["label"].astype(int).tolist()
    # Preserve the human-readable original_label for auditability.
    original_labels = combined["original_label"].astype(str).tolist()
    # Preserve the corpus name so source_counts in later reports still work.
    sources = combined["source"].astype(str).tolist()
    # Preserve the schema split column (still unassigned until train_baseline).
    splits = combined["split"].astype(str).tolist()
    # Walk every selected row in lockstep.
    interrupted = False
    try:
        for index, (message_id, text, label, original_label, source, split) in enumerate(
            zip(message_ids, texts, labels, original_labels, sources, splits, strict=True),
            start=1,
        ):
            # Skip rows already completed in the checkpoint when resuming.
            if (source, message_id) in done_keys:
                # Count the skip so progress still moves.
                resumed_skipped += 1
                # Continue with the next source row.
                continue
            # Produce a chat-register line via the LLM (or injected fake).
            result = rewrite_message_llm(text, int(label), generate)
            # Persist urls_json from the harvested original URLs (never invented).
            urls_json = json.dumps(result.urls, ensure_ascii=True)
            # Empty source text is dropped without counting as an LLM failure.
            if result.status == "empty":
                # Count this source row as dropped.
                dropped_empty += 1
                # Record the empty status so resume will not re-prompt it.
                _checkpoint_row(
                    conn,
                    source=source,
                    message_id=message_id,
                    text=None,
                    label=int(label),
                    original_label=original_label,
                    split=split,
                    urls_json=urls_json,
                    status="empty",
                    urls_appended=False,
                    rewrite_method=None,
                )
                # Continue with the next source row.
                continue
            # After retry+fallback, skip only when no usable DM could be stored.
            if result.status != "ok" or result.text is None:
                # Count this source row as an LLM/fallback failure.
                llm_failed += 1
                # Record the failure so a later --resume can retry it.
                _checkpoint_row(
                    conn,
                    source=source,
                    message_id=message_id,
                    text=None,
                    label=int(label),
                    original_label=original_label,
                    split=split,
                    urls_json=urls_json,
                    status="llm_failed",
                    urls_appended=False,
                    rewrite_method=None,
                )
                # Continue with the next source row.
                continue
            # Confirm original URLs still appear after Python post-conditions.
            if result.urls and all(url in result.text for url in result.urls):
                # Increment the urls_kept row counter for the log.
                urls_kept_rows += 1
            # Count rows where the post-condition attached missing URLs.
            if result.urls_appended:
                # Increment the append counter for the log.
                urls_appended_rows += 1
            # Stamp llm_intent_v1 or rule_based_v1_fallback from the rewriter.
            method = result.rewrite_method or REWRITE_METHOD
            # Count LLM successes separately from refusal fallbacks.
            if method == FALLBACK_REWRITE_METHOD:
                # This labeled row was kept without storing a Llama refusal.
                llm_refused_then_fallback += 1
            else:
                # The model produced a real DM after at most one research retry.
                llm_ok += 1
            # Persist the successful rewrite (label copied unchanged).
            _checkpoint_row(
                conn,
                source=source,
                message_id=message_id,
                text=result.text,
                label=int(label),
                original_label=original_label,
                split=split,
                urls_json=urls_json,
                status="ok",
                urls_appended=result.urls_appended,
                rewrite_method=method,
            )
            # Count newly completed ok rows for the progress line.
            newly_ok += 1
            # Print running counters without any message content.
            if progress_every > 0 and index % progress_every == 0:
                # Show in / llm_ok / fallback / urls_kept / urls_appended / dropped / failed.
                print(
                    f"progress {index}/{rows_in}  in={rows_in}  "
                    f"llm_ok={llm_ok}  fallback={llm_refused_then_fallback}  "
                    f"urls_kept={urls_kept_rows}  urls_appended={urls_appended_rows}  "
                    f"dropped_empty={dropped_empty}  llm_failed={llm_failed}"
                )
    except KeyboardInterrupt:
        # Keep the checkpoint; the operator can re-run with --resume.
        interrupted = True
        # Print without message content so a crash dump cannot leak corpus text.
        print("Interrupted; checkpoint saved. Re-run with --resume.")
    # Write per-source CSVs from the checkpoint (includes resumed ok rows).
    before_dedup, rows_out = write_output_csvs(conn, output_dir)
    # Close the checkpoint connection after the final CSV dump.
    conn.close()
    # Count how many duplicate rewrites were removed.
    dedup_dropped = before_dedup - rows_out
    # Package the global log payload.
    return {
        "rewrite_method": REWRITE_METHOD,
        "ollama_model": resolved_model,
        "ollama_host": ollama_host,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint),
        "sample_mode": sample_mode,
        "rows_available": rows_available,
        "rows_in": rows_in,
        "limit": limit,
        "stratified_sample": stratified_sample,
        "random_state": random_state,
        "resume": resume,
        "resumed_skipped": resumed_skipped,
        "dropped_empty": dropped_empty,
        "llm_ok": llm_ok,
        "llm_refused_then_fallback": llm_refused_then_fallback,
        "llm_failed": llm_failed,
        "urls_kept_rows": urls_kept_rows,
        "urls_appended_rows": urls_appended_rows,
        "newly_ok": newly_ok,
        "rows_out_before_dedup": before_dedup,
        "dedup_dropped": dedup_dropped,
        "rows_out": rows_out,
        "num_predict": OLLAMA_NUM_PREDICT,
        "llm_max_rewrite_chars": LLM_MAX_REWRITE_CHARS,
        "fallback_rewrite_method": FALLBACK_REWRITE_METHOD,
        "chat_eval_touched": False,
        "complete": not interrupted,
        "live_url_reputation": False,
        "cloud_llm": False,
        "runtime_notes": (
            "Default model is llama3.2:latest (Llama 3.2 3B instruct, already "
            "pulled). Local Ollama on WSL2 only. num_predict=400 so a ~600-char "
            "DM plus URLs is not truncated. After GPU warmup ~0.4–0.8s/row on "
            "an RTX 4060 8GB; refusals cost a second generate then rule_based "
            "fallback. Full processed corpora (~71k rows) ≈ 8–12 hours. "
            "Use --resume after a crash. --stratified-sample 10000 ≈ 1–2 hours."
        ),
    }


# Run the full rewrite and persist the row-count log.
def main() -> None:
    """Rewrite processed corpora into data/processed_chat_llm and print counts."""

    # Parse CLI arguments (defaults match the documented from-ml/ command).
    args = parse_args()
    # Record wall-clock time so the log can document runtime.
    started = time.perf_counter()
    # Rewrite every selected processed row into intent-preserving DM text.
    log = rewrite_corpora(
        args.input_dir,
        args.output_dir,
        generate=None,
        model=args.model,
        ollama_host=args.ollama_host,
        timeout=args.timeout,
        resume=args.resume,
        limit=args.limit,
        stratified_sample=args.stratified_sample,
        random_state=args.random_state,
        progress_every=args.progress_every,
        checkpoint_path=args.checkpoint_path,
    )
    # Store elapsed seconds on the log payload.
    log["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    # Ensure the reports directory exists before writing the log.
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist the in/dropped/urls_kept/llm_failed log as JSON for audits.
    args.log_path.write_text(json.dumps(log, indent=2))
    # Print a concise human-readable summary without message content.
    print(
        f"in: {log['rows_in']}  llm_ok: {log['llm_ok']}  "
        f"fallback: {log['llm_refused_then_fallback']}  "
        f"urls_kept: {log['urls_kept_rows']}  "
        f"urls_appended: {log['urls_appended_rows']}  "
        f"dropped_empty: {log['dropped_empty']}  "
        f"llm_failed: {log['llm_failed']}  "
        f"dedup_dropped: {log['dedup_dropped']}  out: {log['rows_out']}"
    )
    # Point the operator at the output directory.
    print(
        f"Wrote LLM chat-register CSVs under {args.output_dir} "
        f"({REWRITE_METHOD}, model={log['ollama_model']})"
    )
    # Point the operator at the JSON log.
    print(f"Wrote {args.log_path}")


# Run the rewriter only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
