# Refresh frontend/public/ml without re-exporting DistilBERT: drop fp32, write gzip/brotli.

# Import argparse so the operator can point at a non-default public/ml folder.
import argparse

# Import sys so a missing public tree exits non-zero.
import sys

# Import Path to resolve repo-relative frontend/public/ml.
from pathlib import Path

# Import the export helpers that own the skip-list and compression format.
from secure_chat_ml.onnx_export import compress_onnx_tree, strip_fp32_from_browser_tree

# This script lives at ml/scripts/; the repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Default Vite public ML folder (gitignored; populated by export_onnx_web.py).
_DEFAULT_PUBLIC_ML = _REPO_ROOT / "frontend" / "public" / "ml"


# Parse CLI flags for an optional public/ml override.
def parse_args() -> argparse.Namespace:
    """Return parsed arguments for prepare_browser_onnx.py."""

    # Build the parser shown in --help.
    parser = argparse.ArgumentParser(
        description=(
            "Remove model.fp32.onnx from frontend/public/ml and write gzip/brotli "
            "siblings for each serving ONNX. Does not re-quantize DistilBERT."
        )
    )
    # Allow tests or operators to point at a temp tree.
    parser.add_argument(
        "--frontend-public-ml",
        type=Path,
        default=_DEFAULT_PUBLIC_ML,
        help="Vite public/ml directory (default: repo frontend/public/ml).",
    )
    # Parse argv.
    return parser.parse_args()


# Strip fp32, compress serving graphs, print a summary.
def main() -> int:
    """Return 0 on success; 1 when the public tree is missing."""

    # Read the destination folder.
    args = parse_args()
    # Resolve so logs show an absolute path.
    public_ml = args.frontend_public_ml.resolve()
    # Fail clearly when export_onnx_web has never been run.
    if not public_ml.is_dir():
        # Tell the operator to export first.
        print(f"prepare_browser_onnx: missing {public_ml}", file=sys.stderr)
        # Non-zero so CI/scripts can detect the miss.
        return 1
    # Delete leftover 256 MiB fp32 graphs from older copies.
    removed = strip_fp32_from_browser_tree(public_ml)
    # Write .gz and .br beside every serving ONNX (int8 DistilBERT, LSTM, TF-IDF).
    compressed = compress_onnx_tree(public_ml)
    # Print removed fp32 paths so the operator can confirm they are gone.
    print(f"prepare_browser_onnx: removed {len(removed)} fp32/inferred file(s)")
    # Print each removed path.
    for path in removed:
        # Relative paths stay readable in terminal logs.
        print(f"  removed {path}")
    # Print how many serving graphs now have compressed siblings.
    print(f"prepare_browser_onnx: compressed {len(compressed)} serving ONNX file(s)")
    # Print each compressed serving graph.
    for path in compressed:
        # Show the serving path, not the sibling.
        print(f"  compressed {path}")
    # Success.
    return 0


# Run when invoked as a script (uv run python scripts/prepare_browser_onnx.py).
if __name__ == "__main__":
    # Propagate the exit code to the shell.
    raise SystemExit(main())
