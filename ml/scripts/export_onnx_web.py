"""CLI: export the six Slice-6 ONNX Runtime Web checkpoints.

Does not overwrite ml/reports/*.json. Does not start the 71k LLM rewrite.
Does not change train_baseline.py / train_distilbert.py / train_lstm.py defaults.
Copies int8 ONNX + gzip/brotli siblings into frontend/public/ml; omits model.fp32.onnx.

Usage (from ml/):

    uv run python scripts/export_onnx_web.py
    uv run python scripts/export_onnx_web.py --only tfidf_default
    uv run python scripts/export_onnx_web.py --skip-frontend-copy
    uv run python scripts/export_onnx_web.py --no-quantize
    uv run python scripts/prepare_browser_onnx.py
"""

# Import argparse for the documented operator flags.
import argparse

# Import sys to return a non-zero status when a catalog id is unknown.
import sys

# Import Path to locate ml/ and frontend/public/ml relative to this script.
from pathlib import Path

# Import the orchestrator that writes ONNX + sidecars and copies into Vite public/.
from secure_chat_ml.onnx_export import export_all_checkpoints

# Import the six-way catalog so --only can validate ids.
from secure_chat_ml.onnx_web_catalog import CHECKPOINT_CATALOG

# ml/ is the parent of scripts/.
_ML_ROOT = Path(__file__).resolve().parents[1]

# Repo root is the parent of ml/ (Minor Project-II).
_REPO_ROOT = _ML_ROOT.parent

# Default export tree (gitignored). Never write into reports/.
_DEFAULT_EXPORT_ROOT = _ML_ROOT / "exports" / "onnx_web"

# Vite public destination so ChatScreen can fetch /ml/<id>/manifest.json.
_DEFAULT_FRONTEND_PUBLIC = _REPO_ROOT / "frontend" / "public" / "ml"


# Parse operator flags for a full or partial six-way export.
def parse_args() -> argparse.Namespace:
    """Return CLI arguments for export_onnx_web.py."""

    # Describe the script in --help.
    parser = argparse.ArgumentParser(
        description="Export published + sweep-winner checkpoints for ONNX Runtime Web.",
    )
    # Allow a reviewer to re-export one graph without touching the other five.
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Checkpoint ids to export (default: all six in load order).",
    )
    # Park graphs somewhere other than ml/exports/onnx_web when testing.
    parser.add_argument(
        "--export-root",
        type=Path,
        default=_DEFAULT_EXPORT_ROOT,
        help="Directory for ONNX + JSON sidecars (default: ml/exports/onnx_web).",
    )
    # Skip the Vite copy when running a unit-style export into tmp_path.
    parser.add_argument(
        "--skip-frontend-copy",
        action="store_true",
        help="Do not copy artifacts into frontend/public/ml/.",
    )
    # Optional explicit public/ml destination.
    parser.add_argument(
        "--frontend-public-ml",
        type=Path,
        default=_DEFAULT_FRONTEND_PUBLIC,
        help="Vite public directory that serves /ml/<id>/ (gitignored).",
    )
    # pytest and CPU-only boxes can skip DistilBERT int8 quantization.
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Write DistilBERT fp32 graphs only (skip int8 quantization).",
    )
    # Return the populated namespace.
    return parser.parse_args()


# Export the requested checkpoints and print their folders.
def main() -> int:
    """Run the six-way (or filtered) export; return 0 on success."""

    # Parse flags before touching the GPU or the 257 MB DistilBERT weights.
    args = parse_args()
    # Valid ids are the six catalog keys.
    valid_ids = {str(row["id"]) for row in CHECKPOINT_CATALOG}
    # Optional filter; None means export all six.
    only_ids: set[str] | None = None
    # Validate --only so a typo fails before DistilBERT load.
    if args.only is not None:
        # Build the filter set from CLI tokens.
        only_ids = set(args.only)
        # Reject unknown ids rather than silently exporting nothing.
        unknown = only_ids - valid_ids
        # Fail with a clear list of valid names.
        if unknown:
            # Print valid ids so the operator can copy-paste.
            print(f"Unknown checkpoint id(s): {sorted(unknown)}. Valid: {sorted(valid_ids)}")
            # Non-zero exit for scripts/CI.
            return 1
    # Skip copying into frontend/public when the operator asked.
    frontend_dest = None if args.skip_frontend_copy else args.frontend_public_ml
    # Announce the destination so the operator knows reports/ is untouched.
    print(f"export_onnx_web: ml_root={_ML_ROOT}")
    # Announce the export root (gitignored).
    print(f"export_onnx_web: export_root={args.export_root}")
    # Announce whether Vite public/ml will be updated.
    print(f"export_onnx_web: frontend_public_ml={frontend_dest}")
    # Run the catalog in load order 1..6 (or the filtered subset).
    written = export_all_checkpoints(
        _ML_ROOT,
        args.export_root,
        frontend_public_ml=frontend_dest,
        distilbert_quantize=not args.no_quantize,
        only_ids=only_ids,
    )
    # Summarize so the operator can copy paths into the README table later.
    print(f"export_onnx_web: finished {len(written)} checkpoint(s)")
    # A total miss means DistilBERT and TF-IDF both failed; fail the CLI.
    if not written:
        # Non-zero so a scripted re-export cannot look successful.
        return 1
    # Success when at least one graph is on disk (browser check skips the rest).
    return 0


# Run the CLI only when invoked as a script.
if __name__ == "__main__":
    # Propagate the status code to the shell.
    sys.exit(main())
