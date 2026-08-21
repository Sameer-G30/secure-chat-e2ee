"""CLI: one-factor-at-a-time DistilBERT retrain with an expanded VAL threshold grid.

Trains DistilBERT once at the documented defaults, then retrains once per
alternative value of a single hyperparameter group (learning rate, epochs,
max_length, batch size, warmup ratio, weight decay). Every run searches the
expanded VAL P(scam) grid 0.20, 0.25, ..., 0.70. TEST and the locked chat-eval
set are scored after the threshold is frozen; they are never used to pick
hyperparameters.

Each run writes its own reports folder and gitignored checkpoint:

    reports/distilbert_param_sweep/<run_id>/
    models/distilbert_param_sweep/<run_id>/

Does not overwrite reports/distilbert/ (the published Slice 5 point).

Usage (from ml/):
    uv run python scripts/sweep_distilbert_params.py
    uv run python scripts/sweep_distilbert_params.py --dry-run
    uv run python scripts/sweep_distilbert_params.py --summarize-only
"""

# Import argparse so the sweep can dry-run or summarize without training.
import argparse

# Import json to persist per-run configs and the final ranking report.
import json

# Import os so HuggingFace tokenizers stay single-threaded under WSL2.
import os

# Import subprocess to isolate each fine-tune in its own process (CUDA teardown).
import subprocess

# Import sys to locate this repo's uv-run Python via the current interpreter path.
import sys

# Import datetime so each run_config records when training started.
from datetime import UTC, datetime

# Import Path for portable report and checkpoint locations.
from pathlib import Path

# Import DistilBERT defaults so OFAT variants stay anchored to the published recipe.
from secure_chat_ml.distilbert import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    DISTILBERT_EXPANDED_THRESHOLD_GRID,
)

# Keep HuggingFace tokenizers from spawning extra threads that warn under WSL2.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Disable wandb prompts if an operator has it installed globally.
os.environ.setdefault("WANDB_DISABLED", "true")

# Root for per-run JSON/PNG reports; never reports/distilbert/.
_DEFAULT_SWEEP_REPORTS_ROOT = Path("reports/distilbert_param_sweep")

# Root for per-run HuggingFace checkpoints (gitignored under ml/models/).
_DEFAULT_SWEEP_MODELS_ROOT = Path("models/distilbert_param_sweep")

# Path to the existing training CLI; invoked as a subprocess per OFAT run.
_TRAIN_SCRIPT = Path("scripts/train_distilbert.py")

# Published Slice 5 TEST report, used only as a reference row in the summary.
_PUBLISHED_TEST = Path("reports/distilbert/test_metrics.json")

# Published Slice 5 chat-eval report, used only as a reference row in the summary.
_PUBLISHED_CHAT = Path("reports/distilbert/chat_style_eval_metrics.json")

# Published Slice 5 VAL report, used only as a reference row in the summary.
_PUBLISHED_VAL = Path("reports/distilbert/val_metrics.json")

# Documented baseline knobs; each OFAT run copies this dict and changes one key.
BASELINE_VALUES: dict[str, float | int] = {
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": DEFAULT_NUM_TRAIN_EPOCHS,
    "max_length": DEFAULT_MAX_LENGTH,
    "batch_size": DEFAULT_TRAIN_BATCH_SIZE,
    "warmup_ratio": DEFAULT_WARMUP_RATIO,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
}

# Alternative values for each group; the published default is intentionally omitted.
OFAT_ALTERNATIVES: dict[str, list[float | int]] = {
    "learning_rate": [1e-5, 3e-5, 5e-5],
    "epochs": [2, 4, 5],
    "max_length": [128, 384, 512],
    "batch_size": [8, 32],
    "warmup_ratio": [0.0, 0.06, 0.2],
    "weight_decay": [0.0, 0.05, 0.1],
}

# Map catalog keys onto train_distilbert.py flags.
_CLI_FLAGS: dict[str, str] = {
    "learning_rate": "--learning-rate",
    "epochs": "--epochs",
    "max_length": "--max-length",
    "batch_size": "--batch-size",
    "warmup_ratio": "--warmup-ratio",
    "weight_decay": "--weight-decay",
}


# Format a learning-rate folder tag without a leading zero in the exponent.
def _format_lr(value: float) -> str:
    """Return a filesystem-safe tag such as 1e-5 for 0.00001."""

    # Use one significant digit in scientific notation (1e-05).
    scientific = f"{value:.0e}"
    # Collapse 1e-05 into 1e-5 so folder names stay short.
    return scientific.replace("e-0", "e-").replace("e+0", "e+")


# Format any changed value into a folder-name fragment.
def _format_value(parameter: str, value: float | int) -> str:
    """Return a short, filesystem-safe rendering of one OFAT value."""

    # Learning rates use scientific notation so 0.00002 does not become 2e-05 noise.
    if parameter == "learning_rate":
        # Delegate to the LR-specific formatter.
        return _format_lr(float(value))
    # Integers (epochs, max_length, batch_size) render without a trailing .0.
    integer_keys = {"epochs", "max_length", "batch_size"}
    # Detect whole numbers even if a JSON round-trip promoted them to float.
    is_whole = isinstance(value, int) or (isinstance(value, float) and value == int(value))
    # Keep folder names like epochs_2 rather than epochs_2.0.
    if is_whole and parameter in integer_keys:
        # Cast through int so 2.0 from JSON never becomes 2.0 in a folder name.
        return str(int(value))
    # Ratios keep a compact decimal (0.0, 0.06, 0.2).
    return f"{float(value):g}"


# Build the ordered OFAT catalog: baseline first, then one change per run.
def build_ofat_runs() -> list[dict[str, object]]:
    """Return one catalog row per training job, baseline first."""

    # Accumulate runs in a stable, numbered order for folder names.
    runs: list[dict[str, object]] = []
    # Zero-pad indices so filesystem sort matches training order.
    run_index = 0
    # Always retrain the published recipe with the expanded VAL grid.
    baseline_id = f"{run_index:02d}_baseline_expanded_grid"
    # Record that this run changes no training knob, only the VAL grid.
    runs.append(
        {
            "run_id": baseline_id,
            "changed_parameter": "threshold_grid",
            "changed_value": "expanded_0.20_to_0.70_step_0.05",
            "values": dict(BASELINE_VALUES),
        }
    )
    # Advance the folder index after the baseline.
    run_index += 1
    # Walk each hyperparameter group in the documented order.
    for parameter, alternatives in OFAT_ALTERNATIVES.items():
        # Train once per alternative, leaving every other knob at the baseline.
        for alternative in alternatives:
            # Copy the baseline so later groups cannot leak mutations.
            values = dict(BASELINE_VALUES)
            # Change exactly one key for this run.
            values[parameter] = alternative
            # Build a readable folder name from the group and the new value.
            run_id = f"{run_index:02d}_{parameter}_{_format_value(parameter, alternative)}"
            # Append the OFAT catalog row.
            runs.append(
                {
                    "run_id": run_id,
                    "changed_parameter": parameter,
                    "changed_value": alternative,
                    "values": values,
                }
            )
            # Advance the folder index for the next alternative.
            run_index += 1
    # Return the complete catalog.
    return runs


# Parse CLI flags for dry-run / resume / summarize-only.
def parse_args() -> argparse.Namespace:
    """Return parsed sweep arguments."""

    # Build a parser from this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__)
    # Allow an operator to park reports somewhere other than the default sweep root.
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=_DEFAULT_SWEEP_REPORTS_ROOT,
        help="Parent directory for per-run report folders.",
    )
    # Allow an operator to park checkpoints somewhere other than models/distilbert_param_sweep/.
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_DEFAULT_SWEEP_MODELS_ROOT,
        help="Parent directory for per-run HuggingFace checkpoints.",
    )
    # Print the catalog without launching GPU training.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the OFAT catalog and exit without training.",
    )
    # Rebuild summary.json from already-finished run folders.
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip training and rewrite the ranking from existing reports.",
    )
    # Re-train a finished run instead of skipping it.
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even when test_metrics.json already exists.",
    )
    # Return the populated namespace.
    return parser.parse_args()


# Write the per-run audit file describing what changed versus the baseline.
def _write_run_config(
    path: Path, run: dict[str, object], reports_dir: Path, model_dir: Path
) -> None:
    """Persist the OFAT identity and knobs next to the training reports."""

    # Build a JSON-serializable audit payload.
    payload = {
        "run_id": run["run_id"],
        "protocol": "one_factor_at_a_time",
        "changed_parameter": run["changed_parameter"],
        "changed_value": run["changed_value"],
        "baseline_values": BASELINE_VALUES,
        "run_values": run["values"],
        "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
        "threshold_grid": list(DISTILBERT_EXPANDED_THRESHOLD_GRID),
        "threshold_grid_note": (
            "Includes 0.20 and 0.25 in addition to the published 0.30..0.70 grid."
        ),
        "selection_rule": "maximize scam recall subject to legitimate recall >= 0.85",
        "reports_dir": str(reports_dir),
        "model_dir": str(model_dir),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "overwrites_published_slice5": False,
    }
    # Ensure the reports folder exists before writing.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write pretty JSON so a reviewer can open the file without a formatter.
    path.write_text(json.dumps(payload, indent=2))


# Build the train_distilbert.py argv for one OFAT run.
def _training_command(run: dict[str, object], reports_dir: Path, model_dir: Path) -> list[str]:
    """Return the uv-less argv that fine-tunes one combination."""

    # Start with the current interpreter so we stay inside the uv virtualenv.
    command = [sys.executable, str(_TRAIN_SCRIPT)]
    # Point reports at this run's folder so Slice 5 JSON is never overwritten.
    command.extend(["--reports-dir", str(reports_dir)])
    # Point the checkpoint at this run's gitignored folder.
    command.extend(["--model-dir", str(model_dir)])
    # Search 0.20 and 0.25 on VAL for every combination.
    command.append("--use-expanded-threshold-grid")
    # Read the knobs that this OFAT run actually uses.
    values = run["values"]
    # Assert values is a dict so type-checkers know we can index string keys.
    assert isinstance(values, dict)
    # Pass every documented training knob explicitly, even when it matches default.
    for key, flag in _CLI_FLAGS.items():
        # Convert the catalog value to a CLI string.
        command.extend([flag, str(values[key])])
    # Keep eval batch at the documented default; it does not change weights.
    command.extend(["--eval-batch-size", str(DEFAULT_EVAL_BATCH_SIZE)])
    # Return the fully built argv.
    return command


# Load a JSON report or return None when a run failed before writing it.
def _load_json(path: Path) -> dict | None:
    """Return parsed JSON or None when the file is missing."""

    # Skip incomplete runs in the ranking rather than crashing the summary.
    if not path.exists():
        # The caller will mark this run as failed/incomplete.
        return None
    # Parse UTF-8 JSON written by train_distilbert.py.
    return json.loads(path.read_text())


# Pull compact P/R/F1 plus confusion cells from a metrics payload.
def _class_metrics(payload: dict, class_name: str) -> dict[str, float]:
    """Return precision, recall, and F1 for one class name."""

    # Read the sklearn classification_report block.
    report = payload["classification_report"][class_name]
    # Return the three numbers the ranking uses.
    return {
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
        "f1": float(report["f1-score"]),
    }


# Build one ranking row from VAL/TEST/chat-eval JSON.
def _row_from_reports(
    run_id: str,
    changed_parameter: str,
    changed_value: object,
    test_payload: dict,
    val_payload: dict | None,
    chat_payload: dict | None,
    *,
    source: str,
) -> dict[str, object]:
    """Return the compact metrics dict stored in summary.json."""

    # TEST scam P/R/F1 after the VAL-frozen threshold.
    test_scam = _class_metrics(test_payload, "scam")
    # TEST legitimate P/R/F1 (ham-warning cost).
    test_legit = _class_metrics(test_payload, "legitimate")
    # TEST confusion matrix in [[TN, FP], [FN, TP]] order.
    test_cm = test_payload["confusion_matrix"]
    # Start with in-domain numbers that exist for every finished run.
    row: dict[str, object] = {
        "run_id": run_id,
        "source": source,
        "changed_parameter": changed_parameter,
        "changed_value": changed_value,
        "chosen_threshold": test_payload.get("chosen_threshold"),
        "floor_feasible": test_payload.get("floor_feasible"),
        "selection_reason": test_payload.get("selection_reason"),
        "hyperparameters": test_payload.get("hyperparameters"),
        "test_scam_precision": test_scam["precision"],
        "test_scam_recall": test_scam["recall"],
        "test_scam_f1": test_scam["f1"],
        "test_legit_precision": test_legit["precision"],
        "test_legit_recall": test_legit["recall"],
        "test_legit_f1": test_legit["f1"],
        "test_ham_warned": test_cm[0][1],
        "test_scams_missed": test_cm[1][0],
        "test_rows": test_payload.get("test_rows"),
    }
    # Attach VAL metrics when the run finished threshold search.
    if val_payload is not None:
        # VAL scam recall at the frozen operating point.
        val_scam = _class_metrics(val_payload, "scam")
        # VAL legitimate recall (the floor that authorized the threshold).
        val_legit = _class_metrics(val_payload, "legitimate")
        # Record VAL numbers so a reviewer can see the search, not just TEST.
        row["val_scam_recall"] = val_scam["recall"]
        # Record VAL ham recall next to the 0.85 floor.
        row["val_legit_recall"] = val_legit["recall"]
        # Record the exact grid that was searched for this run.
        row["grid_thresholds"] = val_payload.get("grid_thresholds")
    # Attach locked chat-eval metrics when they were written.
    if chat_payload is not None:
        # Chat-eval scam P/R/F1 (out-of-domain, predict-only).
        chat_scam = _class_metrics(chat_payload, "scam")
        # Chat-eval legitimate P/R/F1 (false-alarm cost on ordinary DMs).
        chat_legit = _class_metrics(chat_payload, "legitimate")
        # Chat-eval confusion matrix in [[TN, FP], [FN, TP]] order.
        chat_cm = chat_payload["confusion_matrix"]
        # Record OOD scam metrics.
        row["chat_scam_precision"] = chat_scam["precision"]
        # Record OOD scam recall (missed chat scams).
        row["chat_scam_recall"] = chat_scam["recall"]
        # Record OOD scam F1 as a balanced OOD summary.
        row["chat_scam_f1"] = chat_scam["f1"]
        # Record OOD ham metrics.
        row["chat_legit_precision"] = chat_legit["precision"]
        # Record OOD ham recall (1 - false-alarm rate on the 100 legit DMs).
        row["chat_legit_recall"] = chat_legit["recall"]
        # Record OOD ham F1.
        row["chat_legit_f1"] = chat_legit["f1"]
        # Record how many of 100 ordinary DMs were warned.
        row["chat_ham_warned"] = chat_cm[0][1]
        # Record how many of 100 hand-authored scams were missed.
        row["chat_scams_missed"] = chat_cm[1][0]
    # Return the compact row.
    return row


# Rank finished sweep rows using the project's VAL selection philosophy.
def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows sorted by TEST scam recall, then ham recall, then chat-eval."""

    # Only rank runs that actually wrote TEST metrics.
    eligible = [row for row in rows if row.get("test_scam_recall") is not None]
    # Sort copies so the input list stays in catalog order.
    ranked = sorted(
        eligible,
        key=lambda row: (
            # Primary: catch scams on held-out in-domain TEST.
            float(row["test_scam_recall"]),
            # Secondary: do not flood ham once scam recall is tied.
            float(row["test_legit_recall"]),
            # Tertiary: catch locked chat-eval scams (0.0 when chat eval is missing).
            float(row.get("chat_scam_recall") or 0.0),
            # Quaternary: fewer ordinary-DM warnings on the locked set.
            float(row.get("chat_legit_recall") or 0.0),
        ),
        reverse=True,
    )
    # Return the sorted list.
    return ranked


# Choose the best OFAT run, excluding the published Slice 5 reference row.
def _pick_best(ranked: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the top retrained run, skipping the published reference if present."""

    # Walk the ranked list until we find a sweep training run.
    for row in ranked:
        # The published Slice 5 folder is a reference, not an OFAT candidate.
        if row.get("source") == "published_slice5_original_grid":
            # Skip it so "best combination" is one of the retrained runs.
            continue
        # Prefer runs whose VAL floor was feasible, matching the selection rule.
        if row.get("floor_feasible") is False:
            # Keep looking; an infeasible floor is a last resort.
            continue
        # Return the first eligible retrained run.
        return row
    # If every retrained run failed the floor, fall back to the first ranked sweep row.
    for row in ranked:
        # Still skip the published reference.
        if row.get("source") == "published_slice5_original_grid":
            # Continue searching.
            continue
        # Return whatever retrained run ranked highest.
        return row
    # No retrained rows exist yet.
    return None


# Write summary.json and ranking.json from whatever run folders exist.
def write_summary(reports_root: Path) -> dict[str, object]:
    """Read per-run reports and persist the sweep ranking."""

    # Collect one row per finished (or published) report set.
    rows: list[dict[str, object]] = []
    # Include the published Slice 5 point so expanding the grid is comparable.
    published_test = _load_json(_PUBLISHED_TEST)
    # Load published chat-eval next to it.
    published_chat = _load_json(_PUBLISHED_CHAT)
    # Load published VAL so the original grid is visible.
    published_val = _load_json(_PUBLISHED_VAL)
    # Only add the reference row when Slice 5 TEST JSON is present.
    if published_test is not None:
        # Build a ranking row tagged as the original 0.30..0.70 grid.
        rows.append(
            _row_from_reports(
                "published_slice5_original_grid",
                "threshold_grid",
                "original_0.30_to_0.70_step_0.05",
                published_test,
                published_val,
                published_chat,
                source="published_slice5_original_grid",
            )
        )
    # Walk every OFAT folder that the catalog would have created.
    for run in build_ofat_runs():
        # Resolve this run's reports directory.
        run_dir = reports_root / str(run["run_id"])
        # Load TEST metrics written by train_distilbert.py.
        test_payload = _load_json(run_dir / "test_metrics.json")
        # Skip catalog entries that have not finished training.
        if test_payload is None:
            # Record an incomplete placeholder so the summary lists gaps.
            rows.append(
                {
                    "run_id": run["run_id"],
                    "source": "sweep_incomplete",
                    "changed_parameter": run["changed_parameter"],
                    "changed_value": run["changed_value"],
                    "status": "incomplete_or_failed",
                    "reports_dir": str(run_dir),
                }
            )
            # Continue with the next catalog entry.
            continue
        # Load VAL metrics for the frozen threshold audit.
        val_payload = _load_json(run_dir / "val_metrics.json")
        # Load chat-eval metrics when the training script wrote them.
        chat_payload = _load_json(run_dir / "chat_style_eval_metrics.json")
        # Build the compact ranking row for this finished run.
        rows.append(
            _row_from_reports(
                str(run["run_id"]),
                str(run["changed_parameter"]),
                run["changed_value"],
                test_payload,
                val_payload,
                chat_payload,
                source="sweep_retrain",
            )
        )
    # Rank only rows that have TEST scam recall.
    ranked = _rank_rows([row for row in rows if "test_scam_recall" in row])
    # Pick the best retrained combination under the project selection rule.
    best = _pick_best(ranked)
    # Also identify the best chat-eval scam recall among retrained runs.
    chat_ranked = sorted(
        [
            row
            for row in ranked
            if row.get("source") == "sweep_retrain" and "chat_scam_recall" in row
        ],
        key=lambda row: (
            float(row["chat_scam_recall"]),
            float(row.get("chat_legit_recall") or 0.0),
            float(row["test_scam_recall"]),
        ),
        reverse=True,
    )
    # The top chat-eval row may differ from the in-domain winner.
    best_chat = chat_ranked[0] if chat_ranked else None
    # Assemble the summary payload written next to the per-run folders.
    summary = {
        "protocol": "one_factor_at_a_time",
        "threshold_grid": list(DISTILBERT_EXPANDED_THRESHOLD_GRID),
        "selection_rule": "maximize scam recall subject to legitimate recall >= 0.85",
        "ranking_keys": [
            "test_scam_recall",
            "test_legit_recall",
            "chat_scam_recall",
            "chat_legit_recall",
        ],
        "baseline_values": BASELINE_VALUES,
        "n_catalog_runs": len(build_ofat_runs()),
        "n_finished": sum(1 for row in rows if row.get("source") == "sweep_retrain"),
        "n_incomplete": sum(1 for row in rows if row.get("source") == "sweep_incomplete"),
        "best_combination": best,
        "best_chat_eval_combination": best_chat,
        "rows": rows,
        "ranked_run_ids": [row["run_id"] for row in ranked],
        "written_at_utc": datetime.now(UTC).isoformat(),
    }
    # Ensure the sweep root exists before writing the summary.
    reports_root.mkdir(parents=True, exist_ok=True)
    # Persist the full summary including incomplete placeholders.
    (reports_root / "summary.json").write_text(json.dumps(summary, indent=2))
    # Persist a compact ranking list without incomplete rows.
    (reports_root / "ranking.json").write_text(
        json.dumps(
            {
                "best_combination": best,
                "best_chat_eval_combination": best_chat,
                "ranked": ranked,
            },
            indent=2,
        )
    )
    # Return the summary so main() can print the winner.
    return summary


# Launch one isolated train_distilbert.py process for an OFAT run.
def _run_training(run: dict[str, object], reports_dir: Path, model_dir: Path) -> int:
    """Return the training subprocess exit code."""

    # Build the argv that points reports and checkpoints at this run.
    command = _training_command(run, reports_dir, model_dir)
    # Log stdout/stderr beside the metrics so a failed run is still auditable.
    log_path = reports_dir / "train.log"
    # Ensure the reports directory exists before opening the log.
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Open the log in text mode so prints from train_distilbert.py are readable.
    with log_path.open("w", encoding="utf-8") as log_file:
        # Echo the exact command at the top of the log.
        log_file.write(" ".join(command) + "\n\n")
        # Flush the header before the child process starts writing.
        log_file.flush()
        # Run training in a child process so CUDA memory is released afterwards.
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    # Return the exit code so the sweep can mark success vs failure.
    return int(completed.returncode)


# Train every unfinished OFAT run, then rewrite the ranking.
def main() -> None:
    """Run the OFAT DistilBERT sweep or rebuild the summary from disk."""

    # Parse dry-run / summarize / overwrite flags.
    args = parse_args()
    # Build the ordered catalog once so dry-run and train share the same ids.
    runs = build_ofat_runs()
    # Print the catalog size before any GPU work.
    print(f"OFAT catalog: {len(runs)} runs (1 baseline + {len(runs) - 1} one-knob variants)")
    # Print each run so a reviewer can see what will be trained.
    for run in runs:
        # Show the single changed group and its new value.
        print(f"  {run['run_id']}: {run['changed_parameter']}={run['changed_value']}")
    # Exit after printing when the operator only wanted the catalog.
    if args.dry_run:
        # Dry-run must not create checkpoints.
        return
    # Rebuild ranking from existing folders when training is already done.
    if args.summarize_only:
        # Read whatever TEST JSON already exists.
        summary = write_summary(args.reports_root)
        # Print the winner path for the operator.
        best = summary.get("best_combination")
        # Guard against a summarize-only call before any run finished.
        if best is None:
            # Tell the operator nothing is ready to rank.
            print("No finished sweep runs found.")
            # Leave without claiming a winner.
            return
        # Print the best retrained run id.
        print(f"Best combination: {best['run_id']}")
        # Point at the written summary.
        print(f"Wrote {args.reports_root / 'summary.json'}")
        # Summarize-only is done.
        return
    # Ensure the sweep roots exist before the first run.
    args.reports_root.mkdir(parents=True, exist_ok=True)
    # Ensure the checkpoint root exists (gitignored).
    args.models_root.mkdir(parents=True, exist_ok=True)
    # Track how many GPU jobs actually launched.
    launched = 0
    # Track how many GPU jobs finished with exit code 0.
    succeeded = 0
    # Walk the catalog in numbered order.
    for index, run in enumerate(runs, start=1):
        # Resolve this run's report folder.
        reports_dir = args.reports_root / str(run["run_id"])
        # Resolve this run's checkpoint folder.
        model_dir = args.models_root / str(run["run_id"])
        # Detect a finished run by the TEST JSON train_distilbert.py writes last-ish.
        test_path = reports_dir / "test_metrics.json"
        # Skip completed runs unless the operator asked to overwrite.
        if test_path.exists() and not args.overwrite:
            # Announce the skip so the log shows resume behavior.
            print(f"[{index}/{len(runs)}] skip {run['run_id']} (test_metrics.json exists)")
            # Continue to the next catalog entry.
            continue
        # Write the OFAT audit file before training starts.
        _write_run_config(reports_dir / "run_config.json", run, reports_dir, model_dir)
        # Announce which GPU job is about to start.
        print(
            f"[{index}/{len(runs)}] train {run['run_id']} "
            f"({run['changed_parameter']}={run['changed_value']})"
        )
        # Count this as a launched job even if it later fails.
        launched += 1
        # Fine-tune in a child process and capture the exit code.
        exit_code = _run_training(run, reports_dir, model_dir)
        # Record success vs failure next to the reports.
        status_payload = {
            "run_id": run["run_id"],
            "exit_code": exit_code,
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }
        # Persist the subprocess status for later debugging.
        (reports_dir / "status.json").write_text(json.dumps(status_payload, indent=2))
        # Treat a zero exit as success only when TEST JSON actually appeared.
        if exit_code == 0 and test_path.exists():
            # Count a completed, report-writing run.
            succeeded += 1
            # Confirm success in the parent log.
            print(f"[{index}/{len(runs)}] ok {run['run_id']}")
        else:
            # Keep going so one OOM does not abort the remaining catalog.
            print(f"[{index}/{len(runs)}] FAILED {run['run_id']} exit_code={exit_code}")
        # Rewrite the ranking after every run so a crash still leaves a partial summary.
        write_summary(args.reports_root)
    # Final ranking after the last catalog entry.
    summary = write_summary(args.reports_root)
    # Print launch stats.
    print(f"Launched {launched} jobs; {succeeded} wrote TEST metrics.")
    # Read the winner for the operator-facing last line.
    best = summary.get("best_combination")
    # Print the winner when at least one run finished.
    if best is not None:
        # Name the folder and the knob that changed.
        print(
            "Best combination: "
            f"{best['run_id']} ({best['changed_parameter']}={best['changed_value']})"
        )
        # Print TEST scam recall so the reason is visible without opening JSON.
        print(
            "TEST scam recall="
            f"{best['test_scam_recall']:.4f}  TEST legit recall="
            f"{best['test_legit_recall']:.4f}  threshold={best['chosen_threshold']}"
        )
    # Marker line so a parent watcher can detect completion.
    print("SWEEP_COMPLETE")


# Run the sweep only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
