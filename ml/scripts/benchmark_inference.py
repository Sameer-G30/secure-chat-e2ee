"""Report p50/p95/p99 inference latency for TF-IDF and any on-disk neural checkpoints.

Writes reports/cascade/benchmark.json. Does not retune thresholds and does not
fit on chat_style_eval_v1.csv.

Usage (from ml/):
    uv run python scripts/benchmark_inference.py
"""

# Import json to persist percentile tables.
import json

# Import statistics for percentile interpolation.
import statistics

# Import time for wall-clock samples.
import time
from pathlib import Path

from secure_chat_ml.baseline import (
    build_pipeline,
    load_chat_style_eval_set,
    load_processed_corpora,
    stratified_split,
)

_OUTPUT = Path("reports/cascade/benchmark.json")
_V1 = Path("data/chat_eval/chat_style_eval_v1.csv")
_V2 = Path("data/chat_eval/chat_style_eval_v2.csv")


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    """Return p50/p95/p99 for a list of millisecond samples."""

    if not samples_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0.0}
    ordered = sorted(samples_ms)
    # statistics.quantiles needs n>=100 samples; otherwise use median / max.
    if len(ordered) >= 100:
        cuts = statistics.quantiles(ordered, n=100)
        p50 = float(cuts[49])
        p95 = float(cuts[94])
        p99 = float(cuts[98])
    else:
        p50 = float(statistics.median(ordered))
        p95 = float(ordered[-1])
        p99 = float(ordered[-1])
    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean": float(statistics.fmean(ordered)),
        "n": float(len(ordered)),
    }


def _time_lstm(texts: list[str]) -> list[float] | None:
    """Time per-message LSTM predict when models/lstm/model.pt exists."""

    weights = Path("models/lstm/model.pt")
    if not weights.exists():
        return None
    from secure_chat_ml.lstm import load_saved_classifier, predict_scam_proba

    model, token_to_id, url_scaler, _hyperparams, _threshold = load_saved_classifier(
        Path("models/lstm")
    )
    samples: list[float] = []
    for text in texts:
        started = time.perf_counter()
        predict_scam_proba(model, [text], token_to_id, url_scaler)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _time_distilbert(texts: list[str]) -> list[float] | None:
    """Time per-message DistilBERT predict when safetensors weights exist."""

    weights = Path("models/distilbert/model.safetensors")
    alt_weights = Path("models/distilbert/pytorch_model.bin")
    if not weights.exists() and not alt_weights.exists():
        return None
    from secure_chat_ml.distilbert import load_saved_classifier, predict_scam_proba

    model, tokenizer = load_saved_classifier(Path("models/distilbert"))
    # Per-message DistilBERT is expensive; 40 DMs still give a p50/p95 picture.
    timed = texts[:40]
    samples: list[float] = []
    for text in timed:
        started = time.perf_counter()
        predict_scam_proba(model, tokenizer, [text])
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _time_tfidf(texts: list[str]) -> list[float]:
    """Fit TF-IDF on TRAIN then time per-message predict_proba on `texts`."""

    combined = load_processed_corpora(Path("data/processed_chat_llm"))
    train_df, _val_df, _test_df = stratified_split(combined)
    pipeline = build_pipeline(C=0.25)
    pipeline.fit(train_df["text"], train_df["label"])
    samples: list[float] = []
    for text in texts:
        started = time.perf_counter()
        pipeline.predict_proba([text])
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def main() -> None:
    """Write latency percentiles for the locked eval texts (predict-only)."""

    texts: list[str] = []
    if _V1.exists():
        texts.extend(load_chat_style_eval_set(_V1)["text"].astype(str).tolist())
    if _V2.exists():
        texts.extend(load_chat_style_eval_set(_V2)["text"].astype(str).tolist())
    if not texts:
        texts = ["hey are we still on for lunch", "send money now to this wallet"]
    tfidf_ms = _time_tfidf(texts)
    lstm_ms = _time_lstm(texts)
    distilbert_ms = _time_distilbert(texts)
    payload = {
        "chat_style_eval_v1_used_for_training": False,
        "chat_style_eval_v1_used_for_tuning": False,
        "arms": {
            "tfidf": {
                "latency_ms": _percentiles(tfidf_ms),
                "ship_cost_note": (
                    "TypeScript vectorizer + small ONNX logistic head (~tens of KiB)."
                ),
            },
            "lstm": {
                "latency_ms": _percentiles(lstm_ms) if lstm_ms is not None else None,
                "skipped": None if lstm_ms is not None else "checkpoint not on disk",
                "ship_cost_note": (
                    "Word BiLSTM ONNX ~13.2 MiB; two-tier cascade remains shippable."
                ),
            },
            "distilbert": {
                "latency_ms": (
                    _percentiles(distilbert_ms) if distilbert_ms is not None else None
                ),
                "skipped": (
                    None if distilbert_ms is not None else "checkpoint not on disk"
                ),
                "ship_cost_note": (
                    "~64.3 MiB; a cascade that escalates here is not a browser default."
                ),
            },
        },
        "n_texts": len(texts),
    }
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {_OUTPUT}")


if __name__ == "__main__":
    main()
