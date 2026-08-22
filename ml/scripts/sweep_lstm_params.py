"""CLI: one-factor-at-a-time word-BiLSTM retrain with an expanded VAL threshold grid.

Trains the word BiLSTM + URL concat once at the documented defaults, then
retrains once per alternative value of a single hyperparameter group
(learning rate, epochs, max_tokens, embed_dim, hidden_size, num_layers,
dropout, max_vocab_size, batch_size, weight_decay, grad_clip, class_weight,
url_features). Every run searches the expanded VAL P(scam) grid
0.20, 0.25, ..., 0.70. After OFAT finishes, two or three multi-knob combo
runs are trained from the best distinct groups. TEST and the locked
chat-eval set are scored after the threshold is frozen; they are never used
to pick hyperparameters.

Each run writes its own reports folder and gitignored checkpoint:

    reports/lstm_param_sweep/<run_id>/
    models/lstm_param_sweep/<run_id>/

Does not overwrite reports/lstm/ (the published word-BiLSTM point).

Usage (from ml/):
    uv run python scripts/sweep_lstm_params.py
    uv run python scripts/sweep_lstm_params.py --dry-run
    uv run python scripts/sweep_lstm_params.py --summarize-only
"""

# Import argparse so the sweep can dry-run or summarize without training.
import argparse

# Import json to persist per-run configs, combo catalogs, and the ranking.
import json

# Import subprocess to isolate each retrain in its own process (CUDA teardown).
import subprocess

# Import sys to locate this repo's uv-run Python via the current interpreter path.
import sys

# Import datetime so each run_config records when training started.
from datetime import UTC, datetime

# Import Path for portable report and checkpoint locations.
from pathlib import Path

# Import LSTM defaults so OFAT variants stay anchored to the published recipe.
from secure_chat_ml.lstm import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_GRAD_CLIP,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_VOCAB_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_URL_FEATURES,
    DEFAULT_WEIGHT_DECAY,
    LSTM_EXPANDED_THRESHOLD_GRID,
)

# Root for per-run JSON/PNG/markdown reports; never reports/lstm/.
_DEFAULT_SWEEP_REPORTS_ROOT = Path("reports/lstm_param_sweep")

# Root for per-run checkpoints (gitignored under ml/models/).
_DEFAULT_SWEEP_MODELS_ROOT = Path("models/lstm_param_sweep")

# Path to the existing training CLI; invoked as a subprocess per run.
_TRAIN_SCRIPT = Path("scripts/train_lstm.py")

# Sidecar listing multi-knob combo runs decided after OFAT.
_COMBO_CATALOG_NAME = "combo_catalog.json"

# Published LSTM TEST report, used only as a reference row in the summary.
_PUBLISHED_TEST = Path("reports/lstm/test_metrics.json")

# Published LSTM chat-eval report, used only as a reference row in the summary.
_PUBLISHED_CHAT = Path("reports/lstm/chat_style_eval_metrics.json")

# Published LSTM VAL report, used only as a reference row in the summary.
_PUBLISHED_VAL = Path("reports/lstm/val_metrics.json")

# Documented baseline knobs; each OFAT run copies this dict and changes one key.
BASELINE_VALUES: dict[str, object] = {
    "embed_dim": DEFAULT_EMBED_DIM,
    "hidden_size": DEFAULT_HIDDEN_SIZE,
    "num_layers": DEFAULT_NUM_LAYERS,
    "dropout": DEFAULT_DROPOUT,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "max_vocab_size": DEFAULT_MAX_VOCAB_SIZE,
    "batch_size": DEFAULT_BATCH_SIZE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "epochs": DEFAULT_NUM_TRAIN_EPOCHS,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "grad_clip": DEFAULT_GRAD_CLIP,
    "class_weight": DEFAULT_CLASS_WEIGHT,
    "url_features": DEFAULT_URL_FEATURES,
}

# Alternative values for each group; the published default is intentionally omitted.
OFAT_ALTERNATIVES: dict[str, list[object]] = {
    "learning_rate": [5e-4, 2e-3, 5e-3],
    "epochs": [3, 5, 6, 8],
    "max_tokens": [64, 192, 256],
    "embed_dim": [64, 256],
    "hidden_size": [64, 256],
    "num_layers": [2, 3],
    "dropout": [0.0, 0.2, 0.5],
    "max_vocab_size": [10_000, 15_000, 50_000],
    "batch_size": [64, 256],
    "weight_decay": [1e-4, 1e-3],
    "grad_clip": [0.5, 2.0],
    "class_weight": ["none"],
    "url_features": [False],
}

# Map catalog keys onto train_lstm.py flags.
_CLI_FLAGS: dict[str, str] = {
    "embed_dim": "--embed-dim",
    "hidden_size": "--hidden-size",
    "num_layers": "--num-layers",
    "dropout": "--dropout",
    "max_tokens": "--max-tokens",
    "max_vocab_size": "--max-vocab-size",
    "batch_size": "--batch-size",
    "learning_rate": "--learning-rate",
    "epochs": "--epochs",
    "weight_decay": "--weight-decay",
    "grad_clip": "--grad-clip",
    "class_weight": "--class-weight",
}

# Integer knobs that should not render as 2.0 in folder names.
_INTEGER_KEYS = {
    "embed_dim",
    "hidden_size",
    "num_layers",
    "max_tokens",
    "max_vocab_size",
    "batch_size",
    "epochs",
}

# Scientific-notation knobs (learning rate and Adam weight decay).
_SCI_KEYS = {"learning_rate", "weight_decay"}


# Format a small float as 5e-4 rather than 0.0005.
def _format_sci(value: float) -> str:
    """Return a filesystem-safe scientific tag such as 5e-4."""

    # Zero stays a plain 0 so weight_decay=0.0 does not become 0e+0.
    if float(value) == 0.0:
        # Keep folder names short.
        return "0"
    # One significant digit in scientific notation (5e-04).
    scientific = f"{float(value):.0e}"
    # Collapse 5e-04 into 5e-4 so folder names stay short.
    return scientific.replace("e-0", "e-").replace("e+0", "e+")


# Format any changed value into a folder-name fragment.
def _format_value(parameter: str, value: object) -> str:
    """Return a short, filesystem-safe rendering of one OFAT value."""

    # Booleans render as true/false rather than True/False from str().
    if isinstance(value, bool):
        # Match the CLI token style used in report.md headings.
        return "true" if value else "false"
    # Learning rates and weight decay use scientific notation.
    if parameter in _SCI_KEYS:
        # Delegate to the scientific formatter.
        return _format_sci(float(value))  # type: ignore[arg-type]
    # Integers render without a trailing .0.
    is_whole = isinstance(value, int) or (
        isinstance(value, float) and float(value) == int(value)
    )
    # Keep folder names like epochs_8 rather than epochs_8.0.
    if is_whole and parameter in _INTEGER_KEYS:
        # Cast through int so JSON round-trips cannot leave a trailing .0.
        return str(int(value))  # type: ignore[arg-type]
    # Other floats keep a compact decimal (0.2, 0.5).
    if isinstance(value, float):
        # :g drops trailing zeros so 0.20 becomes 0.2.
        return f"{float(value):g}"
    # Strings (class_weight) are already filesystem-safe.
    return str(value).replace(" ", "_")


# Build the ordered OFAT catalog: expanded-grid baseline first, then one change.
def build_ofat_runs() -> list[dict[str, object]]:
    """Return one catalog row per one-knob training job, baseline first."""

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
    # Advance the folder index after the expanded-grid baseline.
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
    # Return the complete OFAT catalog.
    return runs


# Pick the best distinct OFAT groups to combine after one-knob retrains finish.
def select_combo_parameter_groups(
    ranked_rows: list[dict[str, object]],
    *,
    baseline_run_id: str = "00_baseline_expanded_grid",
    max_groups: int = 3,
) -> list[tuple[str, object]]:
    """Return up to `max_groups` (parameter, value) pairs in quality order.

    Prefers groups whose best OFAT alternative beat the expanded-grid baseline
    on combined TEST mean. If fewer than two groups beat the baseline, falls
    back to the highest-ranked distinct OFAT groups so a combo still runs.
    """

    # Locate the expanded-grid baseline row for the improvement comparison.
    baseline_mean = None
    # Walk ranked rows until the expanded-grid retrain is found.
    for row in ranked_rows:
        # Match on folder id rather than source, because 00 is a retrain.
        if row.get("run_id") == baseline_run_id and row.get("combined_mean") is not None:
            # Store the baseline combined mean used as the beat-or-not cut.
            baseline_mean = float(row["combined_mean"])
            # Stop after the first match (ids are unique).
            break
    # Keep the best TEST combined-mean row for each OFAT parameter.
    best_by_param: dict[str, dict[str, object]] = {}
    # Walk every ranked row, including those below the baseline.
    for row in ranked_rows:
        # Ignore published-reference rows and incomplete placeholders.
        if row.get("source") != "sweep_retrain":
            # Combo rows are not OFAT groups.
            continue
        # Read which catalog group this row changed.
        parameter = str(row.get("changed_parameter") or "")
        # Skip the expanded-grid baseline itself (it is not a training knob).
        if parameter in {"threshold_grid", "combo", "val_grids", ""}:
            # Continue scanning other rows.
            continue
        # Skip rows that never wrote a combined mean.
        if row.get("combined_mean") is None:
            # Incomplete metrics cannot rank a group.
            continue
        # Keep this row if it is the first or the best so far for this group.
        previous = best_by_param.get(parameter)
        # No previous winner for this parameter yet.
        if previous is None or float(row["combined_mean"]) > float(previous["combined_mean"]):
            # Store the better OFAT alternative as the group's representative.
            best_by_param[parameter] = row
    # Sort groups by combined TEST mean, highest first.
    ordered = sorted(
        best_by_param.values(),
        key=lambda row: float(row["combined_mean"]),
        reverse=True,
    )
    # Groups that beat the expanded-grid baseline are preferred combo ingredients.
    beaters = [
        row
        for row in ordered
        if baseline_mean is not None and float(row["combined_mean"]) > baseline_mean
    ]
    # Need at least two groups to form a combo; otherwise take the top overall.
    picks = beaters if len(beaters) >= 2 else ordered
    # Cap how many knobs a combo may merge (top-2 and top-3 catalogs).
    picks = picks[: max(int(max_groups), 0)]
    # Convert ranking rows into (parameter, value) pairs for build_combo_runs.
    groups: list[tuple[str, object]] = []
    # Walk the selected group representatives in quality order.
    for row in picks:
        # Pair the catalog key with the alternative that won that group.
        groups.append((str(row["changed_parameter"]), row["changed_value"]))
    # Return the ordered combo ingredients.
    return groups


# Build numbered multi-knob catalog rows from selected OFAT groups.
def build_combo_runs(
    groups: list[tuple[str, object]],
    *,
    start_index: int,
) -> list[dict[str, object]]:
    """Return combo catalog rows (top-2, and top-3 when three groups exist)."""

    # Fewer than two groups cannot form a multi-knob combination.
    if len(groups) < 2:
        # Return an empty catalog so the caller can skip combo training.
        return []
    # Accumulate combo rows in numbered order after the OFAT catalog.
    runs: list[dict[str, object]] = []
    # Folder index continues from the last OFAT id (31 after 00..30).
    run_index = int(start_index)
    # Always train the two-knob merge of the best distinct groups.
    combo_sizes = [2]
    # Add a three-knob merge when a third distinct group is available.
    if len(groups) >= 3:
        # Top-3 is the extra "combine a few" run requested after OFAT.
        combo_sizes.append(3)
    # Materialize one catalog row per combo size.
    for size in combo_sizes:
        # Slice the quality-ordered groups down to this combo's width.
        used = groups[:size]
        # Start from the published knobs and apply every selected alternative.
        values = dict(BASELINE_VALUES)
        # Apply each (parameter, value) pair onto the baseline recipe.
        for parameter, value in used:
            # Overwrite exactly the selected keys; others stay at defaults.
            values[parameter] = value
        # Build a folder tag such as epochs_8__max_tokens_256.
        tag = "__".join(
            f"{parameter}_{_format_value(parameter, value)}" for parameter, value in used
        )
        # Zero-pad the numeric prefix so filesystem sort matches training order.
        run_id = f"{run_index:02d}_combo_{tag}"
        # Persist the merged knobs as a JSON object for report.md.
        changed_value = {parameter: value for parameter, value in used}
        # Append the combo catalog row.
        runs.append(
            {
                "run_id": run_id,
                "changed_parameter": "combo",
                "changed_value": changed_value,
                "values": values,
            }
        )
        # Advance the folder index for the next combo size.
        run_index += 1
    # Return the combo catalog.
    return runs


# Load combo catalog rows previously decided after OFAT.
def load_combo_catalog(reports_root: Path) -> list[dict[str, object]]:
    """Return combo runs from combo_catalog.json, or an empty list."""

    # Resolve the sidecar next to summary.json.
    path = reports_root / _COMBO_CATALOG_NAME
    # Missing sidecar means combos have not been proposed yet.
    if not path.exists():
        # Caller should propose combos after OFAT finishes.
        return []
    # Parse the persisted combo catalog.
    payload = json.loads(path.read_text())
    # Accept either a bare list or an object with a "runs" key.
    if isinstance(payload, list):
        # Legacy/test shape: a raw list of catalog rows.
        return payload
    # Production shape: {"runs": [...], "groups": [...]}.
    runs = payload.get("runs") if isinstance(payload, dict) else None
    # Guard against a corrupt sidecar.
    if not isinstance(runs, list):
        # Treat a bad file as "no combos" rather than crashing the summary.
        return []
    # Return the stored combo rows.
    return runs


# Persist combo catalog rows so resume keeps the same folder ids.
def save_combo_catalog(
    reports_root: Path,
    runs: list[dict[str, object]],
    groups: list[tuple[str, object]],
) -> None:
    """Write combo_catalog.json under the sweep reports root."""

    # Ensure the sweep root exists before writing the sidecar.
    reports_root.mkdir(parents=True, exist_ok=True)
    # JSON cannot dump tuples; store groups as objects.
    groups_payload = [
        {"parameter": parameter, "value": value} for parameter, value in groups
    ]
    # Bundle runs plus the ingredient list for auditability.
    payload = {
        "protocol": "post_ofat_combos",
        "groups": groups_payload,
        "runs": runs,
        "written_at_utc": datetime.now(UTC).isoformat(),
    }
    # Write pretty JSON so a reviewer can open the file without a formatter.
    (reports_root / _COMBO_CATALOG_NAME).write_text(
        json.dumps(payload, indent=2, default=str)
    )


# Union OFAT catalog with any persisted combo runs.
def load_all_runs(reports_root: Path) -> list[dict[str, object]]:
    """Return OFAT rows followed by combo rows when the sidecar exists."""

    # Start with the static one-knob catalog.
    runs = build_ofat_runs()
    # Append multi-knob rows decided after OFAT (may be empty).
    runs.extend(load_combo_catalog(reports_root))
    # Return the combined catalog used by ranking and resume.
    return runs


# Parse CLI flags for dry-run / resume / summarize-only / skip-combos.
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
    # Allow an operator to park checkpoints somewhere other than models/lstm_param_sweep/.
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_DEFAULT_SWEEP_MODELS_ROOT,
        help="Parent directory for per-run checkpoints.",
    )
    # Print the OFAT catalog without launching GPU training.
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
    # Allow an OFAT-only pass when combo training is not wanted.
    parser.add_argument(
        "--skip-combos",
        action="store_true",
        help="Skip post-OFAT multi-knob combo retrains.",
    )
    # Return the populated namespace.
    return parser.parse_args()


# Write the per-run audit file describing what changed versus the baseline.
def _write_run_config(
    path: Path, run: dict[str, object], reports_dir: Path, model_dir: Path
) -> None:
    """Persist the OFAT/combo identity and knobs next to the training reports."""

    # Build a JSON-serializable audit payload.
    payload = {
        "run_id": run["run_id"],
        "protocol": (
            "post_ofat_combo" if run["changed_parameter"] == "combo" else "one_factor_at_a_time"
        ),
        "changed_parameter": run["changed_parameter"],
        "changed_value": run["changed_value"],
        "baseline_values": BASELINE_VALUES,
        "run_values": run["values"],
        "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
        "threshold_grid": list(LSTM_EXPANDED_THRESHOLD_GRID),
        "threshold_grid_note": (
            "Includes 0.20 and 0.25 in addition to the published 0.30..0.70 grid."
        ),
        "selection_rule": "maximize scam recall subject to legitimate recall >= 0.85",
        "reports_dir": str(reports_dir),
        "model_dir": str(model_dir),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "overwrites_published_lstm": False,
    }
    # Ensure the reports folder exists before writing.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write pretty JSON so a reviewer can open the file without a formatter.
    path.write_text(json.dumps(payload, indent=2, default=str))


# Append a boolean CLI flag in argparse BooleanOptionalAction form.
def _append_bool_flag(command: list[str], flag: str, enabled: bool) -> None:
    """Append --flag or --no-flag depending on the OFAT boolean."""

    # True uses the positive flag so the child process does not inherit a stale default.
    if enabled:
        # Positive flag matches train_lstm.py's BooleanOptionalAction names.
        command.append(f"--{flag}")
        # Return after appending the positive form.
        return
    # False uses the explicit --no- form so defaults cannot silently stay on.
    command.append(f"--no-{flag}")


# Build the train_lstm.py argv for one OFAT or combo run.
def _training_command(run: dict[str, object], reports_dir: Path, model_dir: Path) -> list[str]:
    """Return the uv-less argv that trains one combination."""

    # Start with the current interpreter so we stay inside the uv virtualenv.
    command = [sys.executable, str(_TRAIN_SCRIPT)]
    # Point reports at this run's folder so published LSTM JSON is never overwritten.
    command.extend(["--reports-dir", str(reports_dir)])
    # Point the checkpoint at this run's gitignored folder.
    command.extend(["--model-dir", str(model_dir)])
    # Search 0.20 and 0.25 on VAL for every combination.
    command.append("--use-expanded-threshold-grid")
    # Read the knobs that this run actually uses.
    values = run["values"]
    # Assert values is a dict so type-checkers know we can index string keys.
    assert isinstance(values, dict)
    # Pass every documented training knob explicitly, even when it matches default.
    for key, flag in _CLI_FLAGS.items():
        # Convert the catalog value to a CLI string.
        command.extend([flag, str(values[key])])
    # Keep eval batch at the documented default; it does not change weights.
    command.extend(["--eval-batch-size", str(DEFAULT_EVAL_BATCH_SIZE)])
    # URL concat on/off via BooleanOptionalAction.
    _append_bool_flag(command, "url-features", bool(values["url_features"]))
    # Return the fully built argv.
    return command


# Load a JSON report or return None when a run failed before writing it.
def _load_json(path: Path) -> dict | None:
    """Return parsed JSON or None when the file is missing."""

    # Skip incomplete runs in the ranking rather than crashing the summary.
    if not path.exists():
        # The caller will mark this run as failed/incomplete.
        return None
    # Parse UTF-8 JSON written by train_lstm.py.
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
    # Overall TEST accuracy lives at the top of sklearn's output_dict report.
    test_accuracy = float(test_payload["classification_report"]["accuracy"])
    # Combined TEST score matches the DistilBERT / TF-IDF sweep README ranking.
    combined_mean = (test_scam["recall"] + test_legit["precision"] + test_accuracy) / 3.0
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
        "test_accuracy": test_accuracy,
        "combined_mean": combined_mean,
        "test_ham_warned": test_cm[0][1],
        "test_scams_missed": test_cm[1][0],
        "test_rows": test_payload.get("test_rows"),
        "train_wall_clock_seconds": test_payload.get("train_wall_clock_seconds"),
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


# Rank finished sweep rows using the DistilBERT/TF-IDF combined TEST score.
def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows sorted by combined TEST mean, then scam recall, then ham recall."""

    # Only rank runs that actually wrote TEST metrics.
    eligible = [row for row in rows if row.get("test_scam_recall") is not None]
    # Sort copies so the input list stays in catalog order.
    ranked = sorted(
        eligible,
        key=lambda row: (
            # Primary: equal-weight mean of scam recall, ham precision, accuracy.
            float(row.get("combined_mean") or 0.0),
            # Secondary: catch scams on held-out in-domain TEST.
            float(row["test_scam_recall"]),
            # Tertiary: do not flood ham once scam recall is tied.
            float(row["test_legit_recall"]),
            # Quaternary: catch locked chat-eval scams (0.0 when chat eval is missing).
            float(row.get("chat_scam_recall") or 0.0),
            # Quinary: fewer ordinary-DM warnings on the locked set.
            float(row.get("chat_legit_recall") or 0.0),
        ),
        reverse=True,
    )
    # Return the sorted list.
    return ranked


# Choose the best retrained run, excluding the published original-grid reference.
def _pick_best(ranked: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the top retrained run, skipping the published reference if present."""

    # Walk the ranked list until we find a sweep training run with a feasible VAL floor.
    for row in ranked:
        # The published original-grid folder is a reference, not a sweep candidate.
        if row.get("source") == "published_lstm_original_grid":
            # Skip it so "best combination" is one of the retrained runs.
            continue
        # Prefer runs whose VAL floor was feasible, matching the selection rule.
        if row.get("floor_feasible") is False:
            # Keep looking; an infeasible floor is a last resort.
            continue
        # Return the first eligible retrained run (OFAT or combo).
        return row
    # If every retrained run failed the floor, fall back to the first ranked sweep row.
    for row in ranked:
        # Still skip the published reference.
        if row.get("source") == "published_lstm_original_grid":
            # Continue searching.
            continue
        # Return whatever retrained run ranked highest.
        return row
    # No retrained rows exist yet.
    return None


# Write a short human-readable report.md next to the JSON metrics.
def _write_run_markdown(
    reports_dir: Path,
    run: dict[str, object],
    test_payload: dict,
    val_payload: dict | None,
    chat_payload: dict | None,
) -> None:
    """Write report.md describing knobs and TEST/VAL/chat-eval performance."""

    # TEST scam metrics after the VAL-frozen operating point.
    test_scam = _class_metrics(test_payload, "scam")
    # TEST legitimate metrics (false-alarm cost on in-domain ham).
    test_legit = _class_metrics(test_payload, "legitimate")
    # TEST confusion matrix in [[TN, FP], [FN, TP]] order.
    test_cm = test_payload["confusion_matrix"]
    # Overall TEST accuracy from sklearn's classification_report.
    test_accuracy = float(test_payload["classification_report"]["accuracy"])
    # Combined mean used to rank this sweep (same formula as DistilBERT README).
    combined_mean = (test_scam["recall"] + test_legit["precision"] + test_accuracy) / 3.0
    # Render the single changed group or the combo object for the heading.
    changed = f"{run['changed_parameter']} = {run['changed_value']}"
    # Combo runs merge several OFAT winners; OFAT runs change exactly one knob.
    protocol_line = (
        "Protocol: post-OFAT multi-knob combo retrain."
        if run["changed_parameter"] == "combo"
        else "Protocol: one-factor-at-a-time word-BiLSTM retrain."
    )
    # Start the markdown body with identity and protocol.
    lines = [
        f"# {run['run_id']}",
        "",
        protocol_line,
        "",
        f"**Changed parameter:** `{changed}`",
        "",
        "VAL searches thresholds 0.20, 0.25, ..., 0.70.",
        "TEST and the locked chat-eval set are scored after the threshold is frozen.",
        "This folder does not overwrite `reports/lstm/`.",
        "",
        "## Frozen operating point (VALIDATION only)",
        "",
        f"- Chosen threshold: `{test_payload.get('chosen_threshold')}`",
        f"- Floor feasible: `{test_payload.get('floor_feasible')}`",
        f"- Selection reason: `{test_payload.get('selection_reason')}`",
        "",
        "## TEST (in-domain rows; scored once)",
        "",
        f"- Scam precision / recall / F1: "
        f"{test_scam['precision']:.4f} / {test_scam['recall']:.4f} / {test_scam['f1']:.4f}",
        f"- Legitimate precision / recall / F1: "
        f"{test_legit['precision']:.4f} / {test_legit['recall']:.4f} / {test_legit['f1']:.4f}",
        f"- Accuracy: {test_accuracy:.4f}",
        f"- Combined mean (scam recall, ham precision, accuracy): {combined_mean:.4f}",
        f"- Scams missed / ham warned: {test_cm[1][0]} / {test_cm[0][1]}",
        f"- Confusion matrix `[[TN, FP], [FN, TP]]`: `{test_cm}`",
        "",
        "## Training knobs",
        "",
    ]
    # Dump every knob so a reviewer does not have to open run_config.json.
    values = run["values"]
    # Assert values is a dict for type-checkers.
    assert isinstance(values, dict)
    # One bullet per documented training knob.
    for key in BASELINE_VALUES:
        # Show the value that actually ran.
        lines.append(f"- `{key}`: `{values[key]}`")
    # Blank line after the knob list.
    lines.append("")
    # Attach VAL numbers when the training script wrote them.
    if val_payload is not None:
        # VAL scam/ham recall at the frozen point.
        val_scam = _class_metrics(val_payload, "scam")
        # VAL legitimate recall next to the 0.85 floor.
        val_legit = _class_metrics(val_payload, "legitimate")
        # Document the searched grid so the report is auditable without JSON.
        lines.extend(
            [
                "## VALIDATION (threshold search; not reported as the final number)",
                "",
                f"- Scam recall: {val_scam['recall']:.4f}",
                f"- Legitimate recall: {val_legit['recall']:.4f}",
                f"- Threshold grid: `{val_payload.get('grid_thresholds')}`",
                "",
            ]
        )
    # Attach locked chat-eval numbers when they were written.
    if chat_payload is not None:
        # Chat-eval scam metrics (out-of-domain, never used to pick knobs).
        chat_scam = _class_metrics(chat_payload, "scam")
        # Chat-eval legitimate metrics (ordinary-DM false alarms).
        chat_legit = _class_metrics(chat_payload, "legitimate")
        # Chat-eval confusion matrix in [[TN, FP], [FN, TP]] order.
        chat_cm = chat_payload["confusion_matrix"]
        # Document that the 200-row file was predict-only.
        lines.extend(
            [
                "## Locked chat-style eval (200 rows; predict-only; not used for ranking)",
                "",
                f"- Scam precision / recall / F1: "
                f"{chat_scam['precision']:.4f} / {chat_scam['recall']:.4f} / "
                f"{chat_scam['f1']:.4f}",
                f"- Legitimate precision / recall / F1: "
                f"{chat_legit['precision']:.4f} / {chat_legit['recall']:.4f} / "
                f"{chat_legit['f1']:.4f}",
                f"- Scams missed / ham warned: {chat_cm[1][0]} / {chat_cm[0][1]}",
                f"- Confusion matrix `[[TN, FP], [FN, TP]]`: `{chat_cm}`",
                "",
            ]
        )
    # Write UTF-8 markdown next to the JSON reports.
    (reports_dir / "report.md").write_text("\n".join(lines))


# Write a sweep-level README.md ranking table next to summary.json.
def _write_index_markdown(
    reports_root: Path,
    summary: dict[str, object],
    ranked: list[dict[str, object]],
) -> None:
    """Write reports/lstm_param_sweep/README.md with the ranking table."""

    # Read the winner for the lead paragraph.
    best = summary.get("best_combination")
    # Read the best chat-eval row when present.
    best_chat = summary.get("best_chat_eval_combination")
    # Start the index with protocol and VAL grid.
    lines = [
        "# Word BiLSTM one-at-a-time parameter sweep",
        "",
        "Offline experiment. Does **not** overwrite `reports/lstm/`.",
        "",
        "Same 71,370-row `llm_intent_v1` corpus, same 70/20/10 split "
        "(`random_state=42`), same VAL rule (maximize scam recall subject to "
        "legitimate recall ≥ 0.85). Every retrain searched VAL thresholds "
        "**0.20, 0.25, …, 0.70**.",
        "",
        "One-factor-at-a-time first (exactly one training knob vs the published "
        "recipe). After those retrains, two or three **combo** runs merge the "
        "best distinct groups.",
        "",
        f"- Catalog OFAT runs: `{summary.get('n_ofat_runs')}`",
        f"- Catalog combo runs: `{summary.get('n_combo_runs')}`",
        f"- Finished retrains: `{summary.get('n_finished')}`",
        "",
    ]
    # Name the TEST ranking winner when at least one run finished.
    if isinstance(best, dict):
        # Lead with the folder a reviewer should open first.
        lines.extend(
            [
                "## Best combination (TEST combined mean)",
                "",
                f"- Run: `{best.get('run_id')}`",
                f"- Changed: `{best.get('changed_parameter')} = {best.get('changed_value')}`",
                f"- Threshold: `{best.get('chosen_threshold')}`",
                f"- TEST scam recall: `{best.get('test_scam_recall')}`",
                f"- TEST ham precision: `{best.get('test_legit_precision')}`",
                f"- TEST accuracy: `{best.get('test_accuracy')}`",
                f"- Combined mean: `{best.get('combined_mean')}`",
                f"- TEST missed / ham warned: "
                f"`{best.get('test_scams_missed')} / {best.get('test_ham_warned')}`",
                f"- Chat missed / ham warned: "
                f"`{best.get('chat_scams_missed')} / {best.get('chat_ham_warned')}`",
                "",
                "Combined mean = equal-weight mean of TEST scam recall, legitimate "
                "precision, and accuracy. Chat eval was scored after freeze and "
                "was **not** used to pick the ranking.",
                "",
            ]
        )
    # Optionally name a different chat-eval winner.
    if isinstance(best_chat, dict) and (
        not isinstance(best, dict) or best_chat.get("run_id") != best.get("run_id")
    ):
        # Chat-eval ranking can disagree with in-domain TEST.
        lines.extend(
            [
                "## Best chat-eval combination (not used for TEST ranking)",
                "",
                f"- Run: `{best_chat.get('run_id')}`",
                f"- Chat scam recall: `{best_chat.get('chat_scam_recall')}`",
                f"- Chat missed / ham warned: "
                f"`{best_chat.get('chat_scams_missed')} / {best_chat.get('chat_ham_warned')}`",
                "",
            ]
        )
    # Ranking table header.
    lines.extend(
        [
            "## Ranking (TEST combined mean)",
            "",
            "| Rank | Run | Changed | Thr | TEST scam recall | TEST ham precision | "
            "TEST accuracy | Combined mean | TEST missed / ham warned | "
            "Chat missed / ham warned |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    # Emit one table row per ranked retrain (skip incomplete placeholders).
    for index, row in enumerate(ranked, start=1):
        # Skip the published reference later if we want sweep-only; include it.
        changed = f"{row.get('changed_parameter')}={row.get('changed_value')}"
        # Compact TEST error counts for the table.
        test_err = f"{row.get('test_scams_missed')} / {row.get('test_ham_warned')}"
        # Compact chat-eval error counts; may be missing on incomplete rows.
        chat_err = f"{row.get('chat_scams_missed')} / {row.get('chat_ham_warned')}"
        # Format floats that exist on finished rows.
        thr = row.get("chosen_threshold")
        # Combined mean is the primary ranking key.
        mean = row.get("combined_mean")
        # Build a markdown table row.
        lines.append(
            f"| {index} | `{row.get('run_id')}` | `{changed}` | `{thr}` | "
            f"{float(row.get('test_scam_recall') or 0):.4f} | "
            f"{float(row.get('test_legit_precision') or 0):.4f} | "
            f"{float(row.get('test_accuracy') or 0):.4f} | "
            f"{float(mean or 0):.4f} | {test_err} | {chat_err} |"
        )
    # Trailing newline for POSIX text files.
    lines.append("")
    # Write the sweep index next to summary.json.
    (reports_root / "README.md").write_text("\n".join(lines))


# Write summary.json and ranking.json from whatever run folders exist.
def write_summary(reports_root: Path) -> dict[str, object]:
    """Read per-run reports and persist the sweep ranking."""

    # Collect one row per finished (or published) report set.
    rows: list[dict[str, object]] = []
    # Include the published LSTM point so expanding the grid is comparable.
    published_test = _load_json(_PUBLISHED_TEST)
    # Load published chat-eval next to it.
    published_chat = _load_json(_PUBLISHED_CHAT)
    # Load published VAL so the original grid is visible.
    published_val = _load_json(_PUBLISHED_VAL)
    # Only add the reference row when published TEST JSON is present.
    if published_test is not None:
        # Build a ranking row tagged as the original 0.30..0.70 grid.
        rows.append(
            _row_from_reports(
                "published_lstm_original_grid",
                "threshold_grid",
                "original_0.30_to_0.70_step_0.05",
                published_test,
                published_val,
                published_chat,
                source="published_lstm_original_grid",
            )
        )
    # Walk every OFAT and combo folder that the catalog would have created.
    catalog = load_all_runs(reports_root)
    # Count static OFAT rows for the index README.
    n_ofat = len(build_ofat_runs())
    # Combo rows are whatever the sidecar currently lists.
    n_combo = max(len(catalog) - n_ofat, 0)
    # Walk catalog entries in numbered order.
    for run in catalog:
        # Resolve this run's reports directory.
        run_dir = reports_root / str(run["run_id"])
        # Load TEST metrics written by train_lstm.py.
        test_payload = _load_json(run_dir / "test_metrics.json")
        # Tag combo rows separately from one-knob retrains.
        source = "sweep_combo" if run["changed_parameter"] == "combo" else "sweep_retrain"
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
                source=source,
            )
        )
        # Refresh the human-readable report.md from the latest JSON.
        _write_run_markdown(run_dir, run, test_payload, val_payload, chat_payload)
    # Rank only rows that have TEST scam recall.
    ranked = _rank_rows([row for row in rows if "test_scam_recall" in row])
    # Pick the best retrained combination under the combined TEST score.
    best = _pick_best(ranked)
    # Also identify the best chat-eval scam recall among retrained runs.
    chat_ranked = sorted(
        [
            row
            for row in ranked
            if row.get("source") in {"sweep_retrain", "sweep_combo"}
            and "chat_scam_recall" in row
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
        "protocol": "one_factor_at_a_time_then_post_ofat_combos",
        "threshold_grid": list(LSTM_EXPANDED_THRESHOLD_GRID),
        "selection_rule": "maximize scam recall subject to legitimate recall >= 0.85",
        "ranking_keys": [
            "combined_mean",
            "test_scam_recall",
            "test_legit_recall",
            "chat_scam_recall",
            "chat_legit_recall",
        ],
        "combined_mean_definition": (
            "equal-weight mean of TEST scam recall, legitimate precision, and accuracy"
        ),
        "baseline_values": BASELINE_VALUES,
        "n_ofat_runs": n_ofat,
        "n_combo_runs": n_combo,
        "n_catalog_runs": len(catalog),
        "n_finished": sum(
            1 for row in rows if row.get("source") in {"sweep_retrain", "sweep_combo"}
        ),
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
    (reports_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # Persist a compact ranking list without incomplete rows.
    (reports_root / "ranking.json").write_text(
        json.dumps(
            {
                "best_combination": best,
                "best_chat_eval_combination": best_chat,
                "ranked": ranked,
            },
            indent=2,
            default=str,
        )
    )
    # Write the human-readable sweep index with the ranking table.
    _write_index_markdown(reports_root, summary, ranked)
    # Return the summary so main() can print the winner.
    return summary


# Launch one isolated train_lstm.py process for an OFAT or combo run.
def _run_training(run: dict[str, object], reports_dir: Path, model_dir: Path) -> int:
    """Return the training subprocess exit code."""

    # Build the argv that points reports and checkpoints at this run.
    command = _training_command(run, reports_dir, model_dir)
    # Log stdout/stderr beside the metrics so a failed run is still auditable.
    log_path = reports_dir / "train.log"
    # Ensure the reports directory exists before opening the log.
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Open the log in text mode so prints from train_lstm.py are readable.
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


# Train one catalog row unless TEST JSON already exists.
def _train_catalog_run(
    run: dict[str, object],
    *,
    reports_root: Path,
    models_root: Path,
    overwrite: bool,
    index: int,
    total: int,
) -> bool:
    """Return True when this run wrote test_metrics.json (new or already present)."""

    # Resolve this run's report folder.
    reports_dir = reports_root / str(run["run_id"])
    # Resolve this run's checkpoint folder.
    model_dir = models_root / str(run["run_id"])
    # Detect a finished run by the TEST JSON train_lstm.py writes.
    test_path = reports_dir / "test_metrics.json"
    # Skip completed runs unless the operator asked to overwrite.
    if test_path.exists() and not overwrite:
        # Announce the skip so the log shows resume behavior.
        print(f"[{index}/{total}] skip {run['run_id']} (test_metrics.json exists)")
        # Existing TEST JSON counts as finished for combo proposal.
        return True
    # Write the audit file before training starts.
    _write_run_config(reports_dir / "run_config.json", run, reports_dir, model_dir)
    # Announce which GPU job is about to start.
    print(
        f"[{index}/{total}] train {run['run_id']} "
        f"({run['changed_parameter']}={run['changed_value']})"
    )
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
        # Confirm success in the parent log.
        print(f"[{index}/{total}] ok {run['run_id']}")
        # Rewrite ranking after every run so a crash still leaves a partial summary.
        write_summary(reports_root)
        # Count this as finished.
        return True
    # Keep going so one OOM does not abort the remaining catalog.
    print(f"[{index}/{total}] FAILED {run['run_id']} exit_code={exit_code}")
    # Still rewrite the summary so the incomplete row is visible.
    write_summary(reports_root)
    # This run did not produce TEST metrics.
    return False


# Train every unfinished OFAT run, then combo runs, then rewrite the ranking.
def main() -> None:
    """Run the OFAT LSTM sweep, optional combos, or rebuild the summary from disk."""

    # Parse dry-run / summarize / overwrite / skip-combos flags.
    args = parse_args()
    # Build the ordered OFAT catalog once so dry-run and train share the same ids.
    ofat_runs = build_ofat_runs()
    # Print the catalog size before any GPU work.
    print(
        f"OFAT catalog: {len(ofat_runs)} runs "
        f"(1 expanded-grid baseline + {len(ofat_runs) - 1} one-knob variants)"
    )
    # Print each OFAT run so a reviewer can see what will be trained.
    for run in ofat_runs:
        # Show the single changed group and its new value.
        print(f"  {run['run_id']}: {run['changed_parameter']}={run['changed_value']}")
    # Exit after printing when the operator only wanted the catalog.
    if args.dry_run:
        # Combos are decided after OFAT metrics exist, so dry-run cannot list them.
        print("Combo runs are proposed after OFAT finishes (top-2 and top-3 groups).")
        # Dry-run must not create checkpoints.
        return
    # Rebuild ranking from existing folders when training is already done.
    if args.summarize_only:
        # Read whatever TEST JSON already exists, including combo folders.
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
    # Track how many GPU jobs actually launched versus skipped.
    launched = 0
    # Track how many GPU jobs finished with TEST JSON.
    succeeded = 0
    # Walk the OFAT catalog in numbered order.
    for index, run in enumerate(ofat_runs, start=1):
        # Detect whether we will skip (existing TEST) vs launch.
        test_path = args.reports_root / str(run["run_id"]) / "test_metrics.json"
        # Count a launch only when training will actually run.
        will_launch = args.overwrite or not test_path.exists()
        # Train or skip this OFAT row.
        finished = _train_catalog_run(
            run,
            reports_root=args.reports_root,
            models_root=args.models_root,
            overwrite=args.overwrite,
            index=index,
            total=len(ofat_runs),
        )
        # Count launched jobs that were not skips.
        if will_launch:
            # Increment the launched counter for operator stats.
            launched += 1
        # Count finished TEST JSON, including skipped-already-done runs.
        if finished:
            # Increment the success counter.
            succeeded += 1
    # After OFAT, propose and train combo runs unless the operator skipped them.
    if not args.skip_combos:
        # Reuse a persisted combo catalog on resume so folder ids stay stable.
        combo_path = args.reports_root / _COMBO_CATALOG_NAME
        # Reuse a persisted combo catalog on resume so folder ids stay stable.
        if combo_path.exists():
            # Load previously proposed combo rows (may be an empty list).
            combo_runs = load_combo_catalog(args.reports_root)
        else:
            # Rank finished OFAT rows to pick combo ingredients.
            ofat_summary = write_summary(args.reports_root)
            # Ranked rows include the published reference; combo selection skips it.
            ranked = _rank_rows(
                [row for row in ofat_summary["rows"] if "test_scam_recall" in row]  # type: ignore[index]
            )
            # Pick the best distinct OFAT groups (prefer those that beat run 00).
            groups = select_combo_parameter_groups(ranked)
            # Number combo folders after the last OFAT id.
            combo_runs = build_combo_runs(groups, start_index=len(ofat_runs))
            # Persist the combo catalog even when it is empty so resume does not retry.
            save_combo_catalog(args.reports_root, combo_runs, groups)
            # Print the ingredient list for the operator log.
            print(
                "Combo groups: "
                + (
                    ", ".join(f"{parameter}={value}" for parameter, value in groups)
                    if groups
                    else "(none)"
                )
            )
        # Print each combo run that will be trained.
        for run in combo_runs:
            # Show the merged knobs.
            print(f"  {run['run_id']}: combo={run['changed_value']}")
        # Total for progress labels includes OFAT plus combos.
        total = len(ofat_runs) + len(combo_runs)
        # Walk combo rows with indices continuing after OFAT.
        for offset, run in enumerate(combo_runs, start=1):
            # Progress index is OFAT count plus this combo's offset.
            index = len(ofat_runs) + offset
            # Detect whether this combo will actually launch.
            test_path = args.reports_root / str(run["run_id"]) / "test_metrics.json"
            # Count a launch only when training will actually run.
            will_launch = args.overwrite or not test_path.exists()
            # Train or skip this combo row.
            finished = _train_catalog_run(
                run,
                reports_root=args.reports_root,
                models_root=args.models_root,
                overwrite=args.overwrite,
                index=index,
                total=total,
            )
            # Count launched combo jobs.
            if will_launch:
                # Increment the launched counter.
                launched += 1
            # Count finished combo TEST JSON.
            if finished:
                # Increment the success counter.
                succeeded += 1
    # Final ranking after the last catalog entry.
    summary = write_summary(args.reports_root)
    # Print launch stats.
    print(f"Launched {launched} jobs; {succeeded} have TEST metrics.")
    # Read the winner for the operator-facing last line.
    best = summary.get("best_combination")
    # Print the winner when at least one run finished.
    if best is not None:
        # Name the folder and the knob that changed.
        print(
            "Best combination: "
            f"{best['run_id']} ({best['changed_parameter']}={best['changed_value']})"
        )
        # Print TEST metrics so the reason is visible without opening JSON.
        print(
            "TEST scam recall="
            f"{best['test_scam_recall']:.4f}  TEST ham precision="
            f"{best['test_legit_precision']:.4f}  TEST accuracy="
            f"{best['test_accuracy']:.4f}  combined_mean="
            f"{best['combined_mean']:.4f}  threshold={best['chosen_threshold']}"
        )
    # Marker line so a parent watcher can detect completion.
    print("SWEEP_COMPLETE")


# Run the sweep only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
