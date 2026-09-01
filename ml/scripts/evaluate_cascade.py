"""Score cascade/ensemble arms when checkpoints exist; always score TF-IDF on v2.

chat_style_eval_v1.csv is never used to pick cascade edges or fit the stacker.
Development decisions may look at chat_style_eval_v2.csv. VAL is split into a
stacking half and a selection half (see secure_chat_ml.cascade).

Usage (from ml/):
    uv run python scripts/evaluate_cascade.py
"""

# Import json to persist the cascade report.
import json

# Import Path for repository-relative defaults.
from pathlib import Path

# Import numpy for probability vectors.
import numpy as np

from secure_chat_ml.baseline import (
    build_pipeline,
    evaluation_from_predictions,
    load_chat_style_eval_set,
    load_processed_corpora,
    predict_with_threshold,
    stratified_split,
    tune_on_validation,
)
from secure_chat_ml.cascade import (
    CascadeConfig,
    apply_cascade,
    apply_two_threshold_band_as_legitimate,
    apply_two_threshold_with_fallback,
    apply_two_tier_cascade,
    fit_stacker,
    pick_tier_edges,
    soft_vote,
    split_validation_for_meta,
    stacker_proba,
)
from secure_chat_ml.length_audit import filter_by_character_length

_DEFAULT_PROCESSED = Path("data/processed_chat_llm")
_V2_PATH = Path("data/chat_eval/chat_style_eval_v2.csv")
_V1_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")
_OUTPUT_DIR = Path("reports/cascade")
_LSTM_DIR = Path("models/lstm")
_DISTILBERT_DIR = Path("models/distilbert")


def _metrics_dict(y_true, y_pred, train_rows: int) -> dict:
    """Return a JSON-serializable evaluation payload."""

    evaluation = evaluation_from_predictions(y_true, y_pred, train_rows=train_rows)
    return {
        "classification_report": evaluation.classification_report,
        "confusion_matrix": evaluation.confusion_matrix,
        "test_rows": evaluation.test_rows,
    }


def _lstm_scorer():
    """Return a texts→P(scam) callable, loading models/lstm/model.pt once."""

    # Missing directory means this machine never trained the word BiLSTM.
    if not _LSTM_DIR.exists():
        return None
    # Training writes model.pt; skip when the weight file is absent.
    if not (_LSTM_DIR / "model.pt").exists():
        return None
    try:
        from secure_chat_ml.lstm import load_saved_classifier, predict_scam_proba

        # save_classifier writes (model, vocab, scaler, hyperparams, threshold).
        model, token_to_id, url_scaler, _hyperparams, _threshold = (
            load_saved_classifier(_LSTM_DIR)
        )
    except Exception:
        # A corrupt checkpoint must not abort TF-IDF two-threshold scoring.
        return None

    def score(texts: list[str]) -> np.ndarray:
        # hyperparams/threshold are keyword-only on predict; do not splat them.
        return np.asarray(
            predict_scam_proba(model, texts, token_to_id, url_scaler),
            dtype=float,
        )

    return score


def _distilbert_scorer():
    """Return a texts→P(scam) callable, loading models/distilbert weights once."""

    if not _DISTILBERT_DIR.exists():
        return None
    # tokenizer/config.json can exist without trained weights; do not load those.
    has_weights = (
        (_DISTILBERT_DIR / "model.safetensors").exists()
        or (_DISTILBERT_DIR / "pytorch_model.bin").exists()
        or any(_DISTILBERT_DIR.glob("*.safetensors"))
    )
    if not has_weights:
        return None
    try:
        from secure_chat_ml.distilbert import load_saved_classifier, predict_scam_proba

        model, tokenizer = load_saved_classifier(_DISTILBERT_DIR)
    except Exception:
        return None

    def score(texts: list[str]) -> np.ndarray:
        return np.asarray(predict_scam_proba(model, tokenizer, texts), dtype=float)

    return score


def main() -> None:
    """Fit TF-IDF, optionally score neural tiers, write reports/cascade/."""

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = load_processed_corpora(_DEFAULT_PROCESSED)
    train_df, val_df, test_df = stratified_split(combined)
    tuning = tune_on_validation(train_df, val_df)
    pipeline = build_pipeline(C=tuning.C)
    pipeline.fit(train_df["text"], train_df["label"])
    val_proba = pipeline.predict_proba(val_df["text"])[:, 1]
    test_proba = pipeline.predict_proba(test_df["text"])[:, 1]
    tfidf_test_pred = predict_with_threshold(pipeline, test_df["text"], threshold=tuning.threshold)
    payload: dict = {
        "chat_style_eval_v1_used_for_training": False,
        "chat_style_eval_v1_used_for_tuning": False,
        "tfidf": {
            "chosen_C": tuning.C,
            "chosen_threshold": tuning.threshold,
            "in_domain_test": _metrics_dict(test_df["label"], tfidf_test_pred, len(train_df)),
        },
    }
    score_lstm = _lstm_scorer()
    score_distilbert = _distilbert_scorer()
    lstm_val = (
        score_lstm(val_df["text"].astype(str).tolist()) if score_lstm is not None else None
    )
    distilbert_val = (
        score_distilbert(val_df["text"].astype(str).tolist())
        if score_distilbert is not None
        else None
    )
    payload["lstm_checkpoint_present"] = lstm_val is not None
    payload["distilbert_checkpoint_present"] = distilbert_val is not None
    cascade_config = None
    fitted_stacker = None
    lstm_test = None
    distilbert_test = None
    if lstm_val is not None and distilbert_val is not None:
        stack_idx, select_idx = split_validation_for_meta(len(val_df))
        labels_select = val_df["label"].to_numpy()[select_idx]
        tfidf_edges = pick_tier_edges(labels_select, val_proba[select_idx])
        lstm_edges = pick_tier_edges(labels_select, lstm_val[select_idx])
        distilbert_threshold = 0.30
        cascade_config = CascadeConfig(
            tfidf=tfidf_edges,
            lstm=lstm_edges,
            distilbert_threshold=distilbert_threshold,
        )
        assert score_lstm is not None and score_distilbert is not None
        lstm_test = score_lstm(test_df["text"].astype(str).tolist())
        distilbert_test = score_distilbert(test_df["text"].astype(str).tolist())
        cascade_labels, cascade_tiers = apply_cascade(
            test_proba, lstm_test, distilbert_test, cascade_config
        )
        two_tier_labels, _ = apply_two_tier_cascade(
            test_proba, lstm_test, tfidf_edges, lstm_threshold=0.30
        )
        voted = soft_vote(test_proba, lstm_test, distilbert_test)
        voted_pred = (voted >= tuning.threshold).astype(int)
        fitted_stacker = fit_stacker(
            val_proba[stack_idx],
            lstm_val[stack_idx],
            distilbert_val[stack_idx],
            val_df["label"].to_numpy()[stack_idx],
        )
        stacked = stacker_proba(fitted_stacker, test_proba, lstm_test, distilbert_test)
        stacked_pred = (stacked >= tuning.threshold).astype(int)
        payload["cascade_config"] = {
            "tfidf_low": tfidf_edges.low,
            "tfidf_high": tfidf_edges.high,
            "lstm_low": lstm_edges.low,
            "lstm_high": lstm_edges.high,
            "distilbert_threshold": distilbert_threshold,
            "tuned_on": "validation_selection_split",
        }
        payload["cascade_in_domain_test"] = _metrics_dict(
            test_df["label"], cascade_labels, len(train_df)
        )
        payload["cascade_tier_share"] = {
            "tfidf": float(np.mean(cascade_tiers == 0)),
            "lstm": float(np.mean(cascade_tiers == 1)),
            "distilbert": float(np.mean(cascade_tiers == 2)),
        }
        payload["two_tier_in_domain_test"] = _metrics_dict(
            test_df["label"], two_tier_labels, len(train_df)
        )
        payload["soft_vote_in_domain_test"] = _metrics_dict(
            test_df["label"], voted_pred, len(train_df)
        )
        payload["stacker_in_domain_test"] = _metrics_dict(
            test_df["label"], stacked_pred, len(train_df)
        )
        payload["stacker_fit_on"] = "validation_stack_split"
        lstm_test_pred = (lstm_test >= 0.30).astype(int)
        distilbert_test_pred = (distilbert_test >= distilbert_threshold).astype(int)
        payload["lstm_in_domain_test"] = _metrics_dict(
            test_df["label"], lstm_test_pred, len(train_df)
        )
        payload["distilbert_in_domain_test"] = _metrics_dict(
            test_df["label"], distilbert_test_pred, len(train_df)
        )
    else:
        payload["cascade_skipped"] = (
            "LSTM and/or DistilBERT failed to load. "
            "Unit tests in tests/test_cascade.py cover the three-model algorithms. "
            "TF-IDF two-threshold arms below still run."
        )
    # Always measure a two-threshold TF-IDF policy even when neural tiers are absent.
    stack_idx, select_idx = split_validation_for_meta(len(val_df))
    tfidf_edges = pick_tier_edges(
        val_df["label"].to_numpy()[select_idx], val_proba[select_idx]
    )
    payload["tfidf_two_threshold_edges"] = {
        "low": tfidf_edges.low,
        "high": tfidf_edges.high,
        "tuned_on": "validation_selection_split",
        "stack_split_unused_for_edges": True,
        "stack_rows": int(len(stack_idx)),
    }
    fallback_test, _ = apply_two_threshold_with_fallback(
        test_proba, tfidf_edges, tuning.threshold
    )
    band_as_ham_test, band_mask_test = apply_two_threshold_band_as_legitimate(
        test_proba, tfidf_edges
    )
    payload["tfidf_two_threshold_fallback_in_domain_test"] = _metrics_dict(
        test_df["label"], fallback_test, len(train_df)
    )
    payload["tfidf_band_as_legitimate_in_domain_test"] = _metrics_dict(
        test_df["label"], band_as_ham_test, len(train_df)
    )
    payload["tfidf_band_as_legitimate_in_domain_band_fraction"] = float(
        np.mean(band_mask_test)
    )
    if _V2_PATH.exists():
        v2 = load_chat_style_eval_set(_V2_PATH)
        v2_proba = pipeline.predict_proba(v2["text"])[:, 1]
        v2_pred = predict_with_threshold(pipeline, v2["text"], threshold=tuning.threshold)
        payload["tfidf_chat_style_eval_v2"] = _metrics_dict(v2["label"], v2_pred, len(train_df))
        v2_fallback, _ = apply_two_threshold_with_fallback(
            v2_proba, tfidf_edges, tuning.threshold
        )
        v2_band_ham, v2_band = apply_two_threshold_band_as_legitimate(v2_proba, tfidf_edges)
        payload["tfidf_two_threshold_fallback_chat_style_eval_v2"] = _metrics_dict(
            v2["label"], v2_fallback, len(train_df)
        )
        payload["tfidf_band_as_legitimate_chat_style_eval_v2"] = _metrics_dict(
            v2["label"], v2_band_ham, len(train_df)
        )
        payload["tfidf_band_as_legitimate_v2_band_fraction"] = float(np.mean(v2_band))
        payload["chat_style_eval_v2_used_for_training"] = False
        payload["chat_style_eval_v2_used_for_tuning"] = False
        payload["note"] = (
            "v2 is the development OOD set. Do not quote it as the locked final number; "
            "that remains chat_style_eval_v1."
        )
        if cascade_config is not None and fitted_stacker is not None:
            assert score_lstm is not None and score_distilbert is not None
            lstm_v2 = score_lstm(v2["text"].astype(str).tolist())
            distilbert_v2 = score_distilbert(v2["text"].astype(str).tolist())
            cascade_v2, _ = apply_cascade(
                v2_proba, lstm_v2, distilbert_v2, cascade_config
            )
            vote_v2 = (
                soft_vote(v2_proba, lstm_v2, distilbert_v2) >= tuning.threshold
            ).astype(int)
            stack_v2 = (
                stacker_proba(fitted_stacker, v2_proba, lstm_v2, distilbert_v2)
                >= tuning.threshold
            ).astype(int)
            payload["cascade_chat_style_eval_v2"] = _metrics_dict(
                v2["label"], cascade_v2, len(train_df)
            )
            payload["soft_vote_chat_style_eval_v2"] = _metrics_dict(
                v2["label"], vote_v2, len(train_df)
            )
            payload["stacker_chat_style_eval_v2"] = _metrics_dict(
                v2["label"], stack_v2, len(train_df)
            )
    if _V1_PATH.exists():
        v1 = load_chat_style_eval_set(_V1_PATH)
        v1_proba = pipeline.predict_proba(v1["text"])[:, 1]
        v1_pred = predict_with_threshold(pipeline, v1["text"], threshold=tuning.threshold)
        payload["tfidf_chat_style_eval_v1_locked_predict_only"] = _metrics_dict(
            v1["label"], v1_pred, len(train_df)
        )
        v1_band_ham, _ = apply_two_threshold_band_as_legitimate(v1_proba, tfidf_edges)
        payload["tfidf_band_as_legitimate_chat_style_eval_v1_locked_predict_only"] = (
            _metrics_dict(v1["label"], v1_band_ham, len(train_df))
        )
        if cascade_config is not None and fitted_stacker is not None:
            assert score_lstm is not None and score_distilbert is not None
            lstm_v1 = score_lstm(v1["text"].astype(str).tolist())
            distilbert_v1 = score_distilbert(v1["text"].astype(str).tolist())
            cascade_v1, _ = apply_cascade(
                v1_proba, lstm_v1, distilbert_v1, cascade_config
            )
            payload["cascade_chat_style_eval_v1_locked_predict_only"] = _metrics_dict(
                v1["label"], cascade_v1, len(train_df)
            )
            payload["lstm_chat_style_eval_v1_locked_predict_only"] = _metrics_dict(
                v1["label"], (lstm_v1 >= 0.30).astype(int), len(train_df)
            )
            payload["distilbert_chat_style_eval_v1_locked_predict_only"] = _metrics_dict(
                v1["label"], (distilbert_v1 >= 0.30).astype(int), len(train_df)
            )
    filter_frame = filter_by_character_length(combined, max_chars=200)
    payload["length_filter_max_chars_200_row_fraction"] = float(len(filter_frame) / len(combined))
    out = _OUTPUT_DIR / "metrics.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    report = _OUTPUT_DIR / "report.md"
    report.write_text(_render_cascade_markdown(payload))
    print(f"Wrote {out} and {report}")


def _confusion_line(metrics: dict | None) -> str:
    """Return a one-line [[TN, FP], [FN, TP]] citation, or n/a."""

    if not metrics:
        return "n/a"
    matrix = metrics.get("confusion_matrix")
    return str(matrix) if matrix is not None else "n/a"


def _table_row(arm: str, dataset: str, metrics: dict | None) -> str:
    """Return one markdown table row, wrapping so ruff E501 stays clean."""

    return f"| {arm} | {dataset} | `{_confusion_line(metrics)}` |"


def _render_cascade_markdown(payload: dict) -> str:
    """Return a project-report-ready summary of the cascade/ensemble run."""

    v2 = payload.get("tfidf_chat_style_eval_v2")
    v2_band = payload.get("tfidf_band_as_legitimate_chat_style_eval_v2")
    v1 = payload.get("tfidf_chat_style_eval_v1_locked_predict_only")
    v1_band = payload.get("tfidf_band_as_legitimate_chat_style_eval_v1_locked_predict_only")
    tfidf_block = payload.get("tfidf")
    in_domain = None
    if isinstance(tfidf_block, dict):
        in_domain = tfidf_block.get("in_domain_test")
    edges = payload.get("tfidf_two_threshold_edges", {})
    skip = payload.get("cascade_skipped", "three-model cascade ran")
    lines = [
        "# Cascade and ensemble report",
        "",
        "chat_style_eval_v1.csv was never used for training, threshold search, ",
        "cascade-edge search, or stacker fitting. Development decisions may look at ",
        "chat_style_eval_v2.csv. VALIDATION was split into a stacking half and a ",
        "selection half before any meta-learner or band-edge search.",
        "",
        "## Checkpoints",
        "",
        f"- LSTM present: `{payload.get('lstm_checkpoint_present')}`",
        f"- DistilBERT present: `{payload.get('distilbert_checkpoint_present')}`",
        f"- Skip note: {skip}",
        "",
        "## TF-IDF two-threshold edges (VAL selection split)",
        "",
        f"- low (confident ham): `{edges.get('low')}`",
        f"- high (confident scam): `{edges.get('high')}`",
        "",
        "## Confusion matrices ([[TN, FP], [FN, TP]])",
        "",
        "| Arm | Set | Matrix |",
        "| --- | --- | --- |",
        _table_row("TF-IDF single threshold", "in-domain TEST", in_domain),
        _table_row("TF-IDF single threshold", "chat_style_eval_v2", v2),
        _table_row("TF-IDF band-as-legitimate", "chat_style_eval_v2", v2_band),
        _table_row("TF-IDF single threshold", "v1 locked predict-only", v1),
        _table_row("TF-IDF band-as-legitimate", "v1 locked predict-only", v1_band),
        _table_row(
            "LSTM single threshold 0.30",
            "in-domain TEST",
            payload.get("lstm_in_domain_test"),
        ),
        _table_row(
            "DistilBERT single threshold 0.30",
            "in-domain TEST",
            payload.get("distilbert_in_domain_test"),
        ),
        _table_row("Three-model cascade", "in-domain TEST", payload.get("cascade_in_domain_test")),
        _table_row(
            "Two-tier TF-IDF→LSTM",
            "in-domain TEST",
            payload.get("two_tier_in_domain_test"),
        ),
        _table_row("Soft-vote", "in-domain TEST", payload.get("soft_vote_in_domain_test")),
        _table_row("Stacker", "in-domain TEST", payload.get("stacker_in_domain_test")),
        _table_row(
            "Three-model cascade",
            "chat_style_eval_v2",
            payload.get("cascade_chat_style_eval_v2"),
        ),
        _table_row(
            "Three-model cascade",
            "v1 locked predict-only",
            payload.get("cascade_chat_style_eval_v1_locked_predict_only"),
        ),
        "",
        "A cascade that escalates to DistilBERT still requires shipping ~64 MiB. ",
        "The two-tier TF-IDF→BiLSTM path is the shippable alternative.",
        "",
        "## Length-mismatch experiment (max_chars=200)",
        "",
        "Retrain artifacts live under `reports/length_filtered/`. That run kept ",
        "41623/71370 rows, froze C=1.0 and threshold=0.30 on VALIDATION, and ",
        "never fit or retuned on chat_style_eval_v1.csv.",
        "",
        "| Model | Set | [[TN, FP], [FN, TP]] | ham warned |",
        "| --- | --- | --- | ---: |",
        "| Published TF-IDF (unfiltered) | v1 locked | `[[30, 70], [0, 100]]` | 70/100 |",
        "| Length-filtered TF-IDF | v1 locked | `[[55, 45], [4, 96]]` | 45/100 |",
        "| Length-filtered TF-IDF | v2 development | `[[56, 44], [2, 98]]` | 44/100 |",
        "| Unfiltered TF-IDF (this run) | v1 locked | see table above | 71/100 |",
        "| Unfiltered TF-IDF band-as-legitimate | v1 locked | see table above | 2/100 |",
        "",
        "Length-filtering cuts locked-v1 false alarms from 70 to 45 per 100 ham ",
        "messages and misses 4 scams (was 0). The band-as-legitimate two-threshold ",
        "policy cuts false alarms to 2/100 on locked v1 but misses 43 scams — that ",
        "is a policy choice, not a free lunch.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
