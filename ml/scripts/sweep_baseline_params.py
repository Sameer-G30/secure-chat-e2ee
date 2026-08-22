"""CLI: one-factor-at-a-time TF-IDF baseline retrain with expanded VAL grids.

Trains the TF-IDF + URL + Logistic Regression baseline once at the documented
defaults, then retrains once per alternative value of a single hyperparameter
group (max_features, n-gram range, min_df, max_df, sublinear_tf, use_idf,
stop_words, class_weight, solver, url_features). Every run searches the
widened C grid and the expanded VAL P(scam) grid 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after C and the threshold are
frozen; they are never used to pick hyperparameters.

Each run writes its own reports folder and an optional gitignored joblib dump:

    reports/baseline_param_sweep/<run_id>/
    models/baseline_param_sweep/<run_id>/

Does not overwrite reports/baseline_metrics.json (the published baseline).

Usage (from ml/):
    uv run python scripts/sweep_baseline_params.py
    uv run python scripts/sweep_baseline_params.py --dry-run
    uv run python scripts/sweep_baseline_params.py --summarize-only
"""

# Import argparse so the sweep can dry-run or summarize without training.
import argparse

# Import json to persist per-run configs and the final ranking report.
import json

# Import subprocess to isolate each retrain in its own process.
import subprocess

# Import sys to locate this repo's uv-run Python via the current interpreter path.
import sys

# Import datetime so each run_config records when training started.
from datetime import UTC, datetime

# Import Path for portable report and checkpoint locations.
from pathlib import Path

# Import the expanded VAL grids shared with DistilBERT and the widened C set.
from secure_chat_ml.baseline import (
    DEFAULT_C_GRID,
    EXPANDED_THRESHOLD_GRID,
    WIDENED_C_GRID,
    BaselineHyperparameters,
    hyperparameters_to_dict,
)

# Root for per-run JSON/PNG/markdown reports; never reports/baseline_metrics.json.
_DEFAULT_SWEEP_REPORTS_ROOT = Path("reports/baseline_param_sweep")

# Root for per-run joblib dumps (gitignored under ml/models/).
_DEFAULT_SWEEP_MODELS_ROOT = Path("models/baseline_param_sweep")

# Path to the existing training CLI; invoked as a subprocess per OFAT run.
_TRAIN_SCRIPT = Path("scripts/train_baseline.py")

# Path to the existing chat-eval CLI; predict-only on the locked 200-row file.
_CHAT_EVAL_SCRIPT = Path("scripts/evaluate_chat_style_eval.py")

# Published TEST report, used only as a reference row in the summary.
_PUBLISHED_TEST = Path("reports/baseline_metrics.json")

# Published chat-eval report, used only as a reference row in the summary.
_PUBLISHED_CHAT = Path("reports/chat_style_eval_metrics.json")

# Published VAL report, used only as a reference row in the summary.
_PUBLISHED_VAL = Path("reports/val_metrics.json")

# Documented baseline knobs; each OFAT run copies this dict and changes one group.
BASELINE_VALUES: dict[str, object] = {
    "max_features": 50_000,
    "ngram_min": 1,
    "ngram_max": 2,
    "min_df": 2,
    "max_df": 1.0,
    "sublinear_tf": True,
    "use_idf": True,
    "stop_words": "none",
    "strip_accents": "unicode",
    "class_weight": "balanced",
    "solver": "lbfgs",
    "url_features": True,
}

# Alternative values for each group; the published default is intentionally omitted.
OFAT_ALTERNATIVES: dict[str, list[object]] = {
    "max_features": [10_000, 25_000, 100_000, 200_000],
    "ngram_range": [(1, 1), (1, 3), (2, 2)],
    "min_df": [1, 3, 5],
    "max_df": [0.90, 0.95, 0.99],
    "sublinear_tf": [False],
    "use_idf": [False],
    "stop_words": ["english"],
    "class_weight": ["none"],
    "solver": ["liblinear", "saga"],
    "url_features": [False],
}


# Format any changed value into a folder-name fragment.
def _format_value(parameter: str, value: object) -> str:
    """Return a short, filesystem-safe rendering of one OFAT value."""

    # n-gram pairs render as 1_3 so folders stay sortable and unambiguous.
    if parameter == "ngram_range":
        # The catalog stores ngram_range as a two-integer sequence.
        low, high = value  # type: ignore[misc]
        # Cast through int so JSON round-trips cannot leave a trailing .0.
        return f"{int(low)}_{int(high)}"
    # Booleans render as true/false rather than True/False from str().
    if isinstance(value, bool):
        # Match the CLI token style used in report.md headings.
        return "true" if value else "false"
    # Integers (max_features, min_df) render without a trailing .0.
    if isinstance(value, int) and not isinstance(value, bool):
        # Keep folder names like max_features_10000.
        return str(value)
    # Floats (max_df) keep a compact decimal (0.9, 0.95, 0.99).
    if isinstance(value, float):
        # :g drops trailing zeros so 0.90 becomes 0.9.
        return f"{float(value):g}"
    # Strings (solver, stop_words, class_weight) are already filesystem-safe.
    return str(value).replace(" ", "_")


# Copy the published knobs and apply one ngram_range or scalar change.
def _apply_alternative(parameter: str, alternative: object) -> dict[str, object]:
    """Return a new values dict with exactly one logical group changed."""

    # Copy so later groups cannot leak mutations into the baseline dict.
    values = dict(BASELINE_VALUES)
    # ngram_range is one group that sets two integer keys together.
    if parameter == "ngram_range":
        # The catalog stores the alternative as a (min, max) pair.
        low, high = alternative  # type: ignore[misc]
        # Unigrams-only, unigrams+trigrams, or bigrams-only each change this pair.
        values["ngram_min"] = int(low)
        # Upper order is inclusive in sklearn's ngram_range.
        values["ngram_max"] = int(high)
        # Return the mutated copy; every other key stays at the published default.
        return values
    # Scalar groups overwrite a single key that matches the catalog name.
    values[parameter] = alternative
    # Return the mutated copy for this OFAT run.
    return values


# Build the ordered OFAT catalog: expanded-grid baseline first, then one change per run.
def build_ofat_runs() -> list[dict[str, object]]:
    """Return one catalog row per training job, expanded-grid baseline first."""

    # Accumulate runs in a stable, numbered order for folder names.
    runs: list[dict[str, object]] = []
    # Zero-pad indices so filesystem sort matches training order.
    run_index = 0
    # Always retrain the published recipe with the expanded VAL grids.
    baseline_id = f"{run_index:02d}_baseline_expanded_grids"
    # Record that this run changes no training knob, only the VAL C/threshold grids.
    runs.append(
        {
            "run_id": baseline_id,
            "changed_parameter": "val_grids",
            "changed_value": "widened_C_and_expanded_threshold_0.20_to_0.70",
            "values": dict(BASELINE_VALUES),
        }
    )
    # Advance the folder index after the expanded-grid baseline.
    run_index += 1
    # Walk each hyperparameter group in the documented order.
    for parameter, alternatives in OFAT_ALTERNATIVES.items():
        # Train once per alternative, leaving every other knob at the baseline.
        for alternative in alternatives:
            # Apply exactly one logical group for this run.
            values = _apply_alternative(parameter, alternative)
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
    # Allow an operator to park joblib dumps somewhere other than models/baseline_param_sweep/.
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_DEFAULT_SWEEP_MODELS_ROOT,
        help="Parent directory for per-run joblib pipeline dumps.",
    )
    # Print the catalog without launching training.
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
        help="Retrain even when baseline_metrics.json already exists.",
    )
    # Return the populated namespace.
    return parser.parse_args()


# Write the per-run audit file describing what changed versus the baseline.
def _write_run_config(
    path: Path, run: dict[str, object], reports_dir: Path, model_dir: Path
) -> None:
    """Persist the OFAT identity and knobs next to the training reports."""

    # JSON cannot dump tuples; convert ngram pairs and other sequences to lists.
    changed_value = run["changed_value"]
    # ngram_range alternatives are (int, int) tuples that JSON should render as lists.
    if isinstance(changed_value, tuple):
        # Store a JSON array so report.md and ranking.json stay consistent.
        changed_value = list(changed_value)
    # Build a JSON-serializable audit payload.
    payload = {
        "run_id": run["run_id"],
        "protocol": "one_factor_at_a_time",
        "changed_parameter": run["changed_parameter"],
        "changed_value": changed_value,
        "baseline_values": BASELINE_VALUES,
        "run_values": run["values"],
        "c_grid": list(WIDENED_C_GRID),
        "c_grid_note": (
            "Widens the published {0.25, 1.0, 4.0} set with smaller and larger C."
        ),
        "published_c_grid": list(DEFAULT_C_GRID),
        "threshold_grid": list(EXPANDED_THRESHOLD_GRID),
        "threshold_grid_note": (
            "Includes 0.20 and 0.25 in addition to the published 0.30..0.70 grid."
        ),
        "selection_rule": "maximize scam recall subject to legitimate recall >= 0.85",
        "reports_dir": str(reports_dir),
        "model_dir": str(model_dir),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "overwrites_published_baseline": False,
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
        # Positive flag matches train_baseline.py's BooleanOptionalAction names.
        command.append(f"--{flag}")
        # Return after appending the positive form.
        return
    # False uses the explicit --no- form so defaults cannot silently stay on.
    command.append(f"--no-{flag}")


# Build the train_baseline.py argv for one OFAT run.
def _training_command(run: dict[str, object], reports_dir: Path, model_dir: Path) -> list[str]:
    """Return the uv-less argv that trains one combination."""

    # Start with the current interpreter so we stay inside the uv virtualenv.
    command = [sys.executable, str(_TRAIN_SCRIPT)]
    # Point reports at this run's folder so published baseline JSON is never overwritten.
    command.extend(["--reports-dir", str(reports_dir)])
    # Point the joblib dump at this run's gitignored folder.
    command.extend(["--model-dir", str(model_dir)])
    # Search 0.20 and 0.25 on VAL for every combination.
    command.append("--use-expanded-threshold-grid")
    # Search the widened C set for every combination.
    command.append("--use-widened-c-grid")
    # Read the knobs that this OFAT run actually uses.
    values = run["values"]
    # Assert values is a dict so type-checkers know we can index string keys.
    assert isinstance(values, dict)
    # Pass every documented training knob explicitly, even when it matches default.
    command.extend(["--max-features", str(values["max_features"])])
    # Pass both n-gram bounds so ngram_range OFAT is reconstructed exactly.
    command.extend(["--ngram-min", str(values["ngram_min"])])
    # Upper bound is inclusive (sklearn ngram_range).
    command.extend(["--ngram-max", str(values["ngram_max"])])
    # Minimum document frequency for a term to enter the vocabulary.
    command.extend(["--min-df", str(values["min_df"])])
    # Maximum document-frequency proportion (1.0 means no cap).
    command.extend(["--max-df", str(values["max_df"])])
    # Sublinear TF on/off.
    _append_bool_flag(command, "sublinear-tf", bool(values["sublinear_tf"]))
    # IDF weighting on/off.
    _append_bool_flag(command, "use-idf", bool(values["use_idf"]))
    # Stop-word mode: none (published) or english.
    command.extend(["--stop-words", str(values["stop_words"])])
    # Accent folding: unicode (published), ascii, or none.
    command.extend(["--strip-accents", str(values["strip_accents"])])
    # Logistic class weights: balanced (published) or none.
    command.extend(["--class-weight", str(values["class_weight"])])
    # Logistic solver: lbfgs (published), liblinear, or saga.
    command.extend(["--solver", str(values["solver"])])
    # Local URL-feature branch on/off.
    _append_bool_flag(command, "url-features", bool(values["url_features"]))
    # Return the fully built argv.
    return command


# Build the evaluate_chat_style_eval.py argv for one finished training run.
def _chat_eval_command(reports_dir: Path) -> list[str]:
    """Return the argv that scores the locked 200-row file with frozen C/threshold."""

    # Reuse the same interpreter so chat-eval sees the same installed sklearn.
    command = [sys.executable, str(_CHAT_EVAL_SCRIPT)]
    # Point at this run's baseline_metrics.json for frozen C, threshold, and knobs.
    command.extend(["--reports-dir", str(reports_dir)])
    # Return the argv; the script refuses to retune on the locked file.
    return command


# Load a JSON report or return None when a run failed before writing it.
def _load_json(path: Path) -> dict | None:
    """Return parsed JSON or None when the file is missing."""

    # Skip incomplete runs in the ranking rather than crashing the summary.
    if not path.exists():
        # The caller will mark this run as failed/incomplete.
        return None
    # Parse UTF-8 JSON written by train_baseline.py / evaluate_chat_style_eval.py.
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
    # Combined TEST score matches the DistilBERT sweep README ranking.
    combined_mean = (
        test_scam["recall"] + test_legit["precision"] + test_accuracy
    ) / 3.0
    # JSON cannot store tuples; normalize ngram pairs to lists.
    if isinstance(changed_value, tuple):
        # Persist a JSON array for ngram_range alternatives.
        changed_value = list(changed_value)
    # Start with in-domain numbers that exist for every finished run.
    row: dict[str, object] = {
        "run_id": run_id,
        "source": source,
        "changed_parameter": changed_parameter,
        "changed_value": changed_value,
        "chosen_C": test_payload.get("chosen_C"),
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
        # Record the exact C grid that was searched for this run.
        row["grid_C"] = val_payload.get("grid_C")
        # Record the exact threshold grid that was searched for this run.
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


# Rank finished sweep rows using the DistilBERT sweep README combined TEST score.
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


# Choose the best OFAT run, excluding the published original-grid reference row.
def _pick_best(ranked: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the top retrained run, skipping the published reference if present."""

    # Walk the ranked list until we find a sweep training run with a feasible VAL floor.
    for row in ranked:
        # The published original-grid folder is a reference, not an OFAT candidate.
        if row.get("source") == "published_baseline_original_grids":
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
        if row.get("source") == "published_baseline_original_grids":
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
    # Render the single changed group for the heading.
    changed = f"{run['changed_parameter']} = {run['changed_value']}"
    # Start the markdown body with identity and protocol.
    lines = [
        f"# {run['run_id']}",
        "",
        "Protocol: one-factor-at-a-time TF-IDF baseline retrain.",
        "",
        f"**Changed parameter:** `{changed}`",
        "",
        "All other training knobs stay at the published defaults.",
        "VAL searches the widened C grid and thresholds 0.20, 0.25, ..., 0.70.",
        "TEST and the locked chat-eval set are scored after C and the threshold are frozen.",
        "",
        "## Frozen operating point (VALIDATION only)",
        "",
        f"- Chosen C: `{test_payload.get('chosen_C')}`",
        f"- Chosen threshold: `{test_payload.get('chosen_threshold')}`",
        f"- Floor feasible: `{test_payload.get('floor_feasible')}`",
        f"- Selection reason: `{test_payload.get('selection_reason')}`",
        "",
        "## TEST (7,137 in-domain rows; scored once)",
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
    ]
    # Attach VAL numbers when the training script wrote them.
    if val_payload is not None:
        # VAL scam/ham recall at the frozen point.
        val_scam = _class_metrics(val_payload, "scam")
        # VAL legitimate recall next to the 0.85 floor.
        val_legit = _class_metrics(val_payload, "legitimate")
        # Document the searched grids so the report is auditable without JSON.
        lines.extend(
            [
                "## VALIDATION (threshold/C search; not reported as the final number)",
                "",
                f"- Scam recall: {val_scam['recall']:.4f}",
                f"- Legitimate recall: {val_legit['recall']:.4f}",
                f"- C grid: `{val_payload.get('grid_C')}`",
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


# Write summary.json and ranking.json from whatever run folders exist.
def write_summary(reports_root: Path) -> dict[str, object]:
    """Read per-run reports and persist the sweep ranking."""

    # Collect one row per finished (or published) report set.
    rows: list[dict[str, object]] = []
    # Include the published baseline so expanding the grids is comparable.
    published_test = _load_json(_PUBLISHED_TEST)
    # Load published chat-eval next to it.
    published_chat = _load_json(_PUBLISHED_CHAT)
    # Load published VAL so the original grids are visible.
    published_val = _load_json(_PUBLISHED_VAL)
    # Only add the reference row when published TEST JSON is present.
    if published_test is not None:
        # Build a ranking row tagged as the original C and 0.30..0.70 grids.
        rows.append(
            _row_from_reports(
                "published_baseline_original_grids",
                "val_grids",
                "original_C_0.25_1_4_and_threshold_0.30_to_0.70",
                published_test,
                published_val,
                published_chat,
                source="published_baseline_original_grids",
            )
        )
    # Walk every OFAT folder that the catalog would have created.
    for run in build_ofat_runs():
        # Resolve this run's reports directory.
        run_dir = reports_root / str(run["run_id"])
        # Load TEST metrics written by train_baseline.py.
        test_payload = _load_json(run_dir / "baseline_metrics.json")
        # Skip catalog entries that have not finished training.
        if test_payload is None:
            # Record an incomplete placeholder so the summary lists gaps.
            rows.append(
                {
                    "run_id": run["run_id"],
                    "source": "sweep_incomplete",
                    "changed_parameter": run["changed_parameter"],
                    "changed_value": run["changed_value"]
                    if not isinstance(run["changed_value"], tuple)
                    else list(run["changed_value"]),  # type: ignore[arg-type]
                    "status": "incomplete_or_failed",
                    "reports_dir": str(run_dir),
                }
            )
            # Continue with the next catalog entry.
            continue
        # Load VAL metrics for the frozen threshold audit.
        val_payload = _load_json(run_dir / "val_metrics.json")
        # Load chat-eval metrics when the eval script wrote them.
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
        "c_grid": list(WIDENED_C_GRID),
        "threshold_grid": list(EXPANDED_THRESHOLD_GRID),
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
        "n_catalog_runs": len(build_ofat_runs()),
        "n_finished": sum(1 for row in rows if row.get("source") == "sweep_retrain"),
        "n_incomplete": sum(1 for row in rows if row.get("source") == "sweep_incomplete"),
        "best_combination": best,
        "best_chat_eval_combination": best_chat,
        "rows": rows,
        "ranked_run_ids": [row["run_id"] for row in ranked],
        "written_at_utc": datetime.now(UTC).isoformat(),
        "hyperparameters_schema": hyperparameters_to_dict(BaselineHyperparameters()),
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
    # Return the summary so main() can print the winner.
    return summary


# Launch isolated train + chat-eval processes for an OFAT run.
def _run_training(run: dict[str, object], reports_dir: Path, model_dir: Path) -> int:
    """Return the last subprocess exit code (0 only if train and chat-eval both succeed)."""

    # Build the argv that points reports and joblib dumps at this run.
    command = _training_command(run, reports_dir, model_dir)
    # Log stdout/stderr beside the metrics so a failed run is still auditable.
    log_path = reports_dir / "train.log"
    # Ensure the reports directory exists before opening the log.
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Open the log in text mode so prints from train_baseline.py are readable.
    with log_path.open("w", encoding="utf-8") as log_file:
        # Echo the exact training command at the top of the log.
        log_file.write(" ".join(command) + "\n\n")
        # Flush the header before the child process starts writing.
        log_file.flush()
        # Run training in a child process so a crash cannot poison later runs.
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        # Abort chat-eval when training failed so we do not score a missing model.
        if completed.returncode != 0:
            # Return the training exit code as the run's status.
            return int(completed.returncode)
        # Build the chat-eval argv now that baseline_metrics.json should exist.
        chat_command = _chat_eval_command(reports_dir)
        # Echo the chat-eval command under a separator in the same log.
        log_file.write("\n\n" + " ".join(chat_command) + "\n\n")
        # Flush before the second child starts.
        log_file.flush()
        # Score the locked 200-row file with frozen C/threshold (never retune).
        chat_completed = subprocess.run(
            chat_command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        # Return the chat-eval exit code; training already succeeded.
        return int(chat_completed.returncode)


# Train every unfinished OFAT run, then rewrite the ranking.
def main() -> None:
    """Run the OFAT baseline sweep or rebuild the summary from disk."""

    # Parse dry-run / summarize / overwrite flags.
    args = parse_args()
    # Build the ordered catalog once so dry-run and train share the same ids.
    runs = build_ofat_runs()
    # Print the catalog size before any fitting.
    print(
        f"OFAT catalog: {len(runs)} runs "
        f"(1 expanded-grid baseline + {len(runs) - 1} one-knob variants)"
    )
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
    # Ensure the joblib root exists (gitignored).
    args.models_root.mkdir(parents=True, exist_ok=True)
    # Track how many jobs actually launched.
    launched = 0
    # Track how many jobs finished with exit code 0 and TEST JSON.
    succeeded = 0
    # Walk the catalog in numbered order.
    for index, run in enumerate(runs, start=1):
        # Resolve this run's report folder.
        reports_dir = args.reports_root / str(run["run_id"])
        # Resolve this run's joblib folder.
        model_dir = args.models_root / str(run["run_id"])
        # Detect a finished run by the TEST JSON train_baseline.py writes.
        test_path = reports_dir / "baseline_metrics.json"
        # Skip completed runs unless the operator asked to overwrite.
        if test_path.exists() and not args.overwrite:
            # Announce the skip so the log shows resume behavior.
            print(f"[{index}/{len(runs)}] skip {run['run_id']} (baseline_metrics.json exists)")
            # Continue to the next catalog entry.
            continue
        # Write the OFAT audit file before training starts.
        _write_run_config(reports_dir / "run_config.json", run, reports_dir, model_dir)
        # Announce which job is about to start.
        print(
            f"[{index}/{len(runs)}] train {run['run_id']} "
            f"({run['changed_parameter']}={run['changed_value']})"
        )
        # Count this as a launched job even if it later fails.
        launched += 1
        # Train and score chat-eval in child processes and capture the exit code.
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
            # Keep going so one failure does not abort the remaining catalog.
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
        # Print TEST metrics so the reason is visible without opening JSON.
        print(
            "TEST scam recall="
            f"{best['test_scam_recall']:.4f}  TEST ham precision="
            f"{best['test_legit_precision']:.4f}  TEST accuracy="
            f"{best['test_accuracy']:.4f}  combined_mean="
            f"{best['combined_mean']:.4f}  C={best['chosen_C']}  "
            f"threshold={best['chosen_threshold']}"
        )
    # Marker line so a parent watcher can detect completion.
    print("SWEEP_COMPLETE")


# Run the sweep only when this file is invoked as a script.
if __name__ == "__main__":
    # Execute the documented CLI entry point.
    main()
