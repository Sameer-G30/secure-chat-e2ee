"""Describe the six Slice-6 ONNX Runtime Web checkpoints and browser fixtures.

This module is data-only: it never trains, never reads the locked chat-eval
CSV for fitting, and never overwrites published metric JSON. Export code
imports these records so the load order stays identical in Python and TS.
"""

# Import Path so catalog entries can be resolved against the ml/ root later.
from pathlib import Path

# Import Any for the loosely typed checkpoint dictionaries.
from typing import Any

# Fixed short DMs used to check banner on/off after every ORT Web load.
# These are not the locked 200-row chat eval set and are never used to tune.
ONNX_WEB_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "ham_no_url",
        "text": "lunch tomorrow? I can do 12:30 if that still works",
        "gold_label": 0,
    },
    {
        "id": "ham_https",
        "text": "here is the shared doc https://docs.google.com/document/d/abc123",
        "gold_label": 0,
    },
    {
        "id": "scam_no_url",
        "text": "mom I need you to buy steam gift cards and text me the codes now",
        "gold_label": 1,
    },
    {
        "id": "scam_shortener",
        "text": "your account is locked verify immediately https://bit.ly/secure-login",
        "gold_label": 1,
    },
)

# Six-way sequential load order. Do not reorder unless a tab crashes.
CHECKPOINT_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "distilbert_best",
        "load_order": 1,
        "family": "distilbert",
        "label": "DistilBERT best (max_length 512, threshold 0.20)",
        "model_dir": "models/distilbert_param_sweep/09_max_length_512",
        "reports_dir": "reports/distilbert_param_sweep/09_max_length_512",
        "metrics_name": "test_metrics.json",
        "threshold": 0.20,
        "max_length": 512,
        "export_dirname": "distilbert_best",
    },
    {
        "id": "distilbert_default",
        "load_order": 2,
        "family": "distilbert",
        "label": "DistilBERT default (max_length 256, threshold 0.30)",
        "model_dir": "models/distilbert",
        "reports_dir": "reports/distilbert",
        "metrics_name": "test_metrics.json",
        "threshold": 0.30,
        "max_length": 256,
        "export_dirname": "distilbert_default",
    },
    {
        "id": "lstm_best",
        "load_order": 3,
        "family": "lstm",
        "label": "Word BiLSTM best (8 epochs, threshold 0.20)",
        "model_dir": "models/lstm_param_sweep/07_epochs_8",
        "reports_dir": "reports/lstm_param_sweep/07_epochs_8",
        "metrics_name": "test_metrics.json",
        "threshold": 0.20,
        "export_dirname": "lstm_best",
    },
    {
        "id": "lstm_default",
        "load_order": 4,
        "family": "lstm",
        "label": "Word BiLSTM default (4 epochs, threshold 0.30)",
        "model_dir": "models/lstm",
        "reports_dir": "reports/lstm",
        "metrics_name": "test_metrics.json",
        "threshold": 0.30,
        "export_dirname": "lstm_default",
    },
    {
        "id": "tfidf_best",
        "load_order": 5,
        "family": "tfidf",
        "label": "TF-IDF best (max_features 10000, C=1.0, threshold 0.20)",
        "model_dir": "models/baseline_param_sweep/01_max_features_10000",
        "reports_dir": "reports/baseline_param_sweep/01_max_features_10000",
        "metrics_name": "baseline_metrics.json",
        "threshold": 0.20,
        "C": 1.0,
        "max_features": 10_000,
        "export_dirname": "tfidf_best",
    },
    {
        "id": "tfidf_default",
        "load_order": 6,
        "family": "tfidf",
        "label": "TF-IDF default (max_features 50000, C=0.25, threshold 0.30)",
        "model_dir": "models/baseline",
        "fallback_fit_dir": "models/baseline_onnx_export",
        "reports_dir": "reports",
        "metrics_name": "baseline_metrics.json",
        "threshold": 0.30,
        "C": 0.25,
        "max_features": 50_000,
        "export_dirname": "tfidf_default",
    },
)


# Resolve a catalog-relative path against the ml/ project root.
def resolve_under_ml(ml_root: Path, relative: str) -> Path:
    """Return ml_root / relative as a resolved Path."""

    # Keep catalog strings relative so tests can point at a fake ml root.
    return (ml_root / relative).resolve()
