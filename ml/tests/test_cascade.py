"""Exercise cascade/ensemble helpers on synthetic probabilities, never live checkpoints."""

# Import numpy to build tiny probability and label vectors.
import numpy as np

from secure_chat_ml.cascade import (
    CascadeConfig,
    TierEdges,
    apply_cascade,
    apply_two_threshold_band_as_legitimate,
    apply_two_threshold_with_fallback,
    apply_two_tier_cascade,
    classify_tier,
    fit_stacker,
    pick_tier_edges,
    soft_vote,
    split_validation_for_meta,
    stacker_proba,
)


def test_classify_tier_settles_confident_rows_and_escalates_the_band() -> None:
    proba = np.asarray([0.05, 0.50, 0.95])
    decision, escalate = classify_tier(proba, TierEdges(low=0.20, high=0.80))
    assert decision.tolist() == [0, -1, 1]
    assert escalate.tolist() == [False, True, False]


def test_apply_cascade_stops_at_the_first_confident_tier() -> None:
    p_tfidf = np.asarray([0.05, 0.50, 0.50])
    p_lstm = np.asarray([0.99, 0.05, 0.50])
    p_distilbert = np.asarray([0.99, 0.99, 0.90])
    config = CascadeConfig(
        tfidf=TierEdges(low=0.20, high=0.80),
        lstm=TierEdges(low=0.20, high=0.80),
        distilbert_threshold=0.50,
    )
    labels, tiers = apply_cascade(p_tfidf, p_lstm, p_distilbert, config)
    assert labels.tolist() == [0, 0, 1]
    assert tiers.tolist() == [0, 1, 2]


def test_two_tier_cascade_never_requires_distilbert() -> None:
    p_tfidf = np.asarray([0.50, 0.05])
    p_lstm = np.asarray([0.90, 0.10])
    labels, tiers = apply_two_tier_cascade(
        p_tfidf, p_lstm, TierEdges(low=0.20, high=0.80), lstm_threshold=0.50
    )
    assert labels.tolist() == [1, 0]
    assert tiers.tolist() == [1, 0]


def test_soft_vote_is_the_arithmetic_mean() -> None:
    voted = soft_vote(
        np.asarray([0.0, 1.0]),
        np.asarray([0.5, 0.5]),
        np.asarray([1.0, 0.0]),
    )
    assert voted.tolist() == [0.5, 0.5]


def test_stacker_fits_only_on_the_provided_split() -> None:
    p_tfidf = np.asarray([0.1, 0.2, 0.8, 0.9])
    p_lstm = np.asarray([0.1, 0.3, 0.7, 0.9])
    p_distilbert = np.asarray([0.2, 0.2, 0.8, 0.8])
    labels = np.asarray([0, 0, 1, 1])
    model = fit_stacker(p_tfidf, p_lstm, p_distilbert, labels)
    proba = stacker_proba(model, p_tfidf, p_lstm, p_distilbert)
    assert proba.shape == (4,)
    assert proba[0] < proba[-1]


def test_split_validation_for_meta_is_disjoint_and_complete() -> None:
    stack_idx, select_idx = split_validation_for_meta(10, random_state=0)
    assert len(stack_idx) + len(select_idx) == 10
    assert set(stack_idx).isdisjoint(set(select_idx))


def test_two_threshold_fallback_uses_the_single_cut_in_the_band() -> None:
    # Three points: confident ham, band, confident scam.
    proba = np.asarray([0.05, 0.40, 0.95])
    # Fallback 0.30 should label the band as scam.
    labels, used = apply_two_threshold_with_fallback(
        proba, TierEdges(low=0.20, high=0.80), fallback_threshold=0.30
    )
    # 0.05 ham, 0.40 >= 0.30 scam, 0.95 scam.
    assert labels.tolist() == [0, 1, 1]
    # Only the middle row used the fallback cut.
    assert used.tolist() == [0, 1, 0]


def test_two_threshold_band_as_legitimate_does_not_warn_in_the_band() -> None:
    # The middle probability is a scam under a 0.30 cut, but not "confident".
    proba = np.asarray([0.05, 0.40, 0.95])
    # Warn only on the high edge; the band is an explicit no-banner decision.
    labels, in_band = apply_two_threshold_band_as_legitimate(
        proba, TierEdges(low=0.20, high=0.80)
    )
    # 0.40 stays legitimate because it is not a confident scam.
    assert labels.tolist() == [0, 0, 1]
    # Only the middle row sat in the escalate band.
    assert in_band.tolist() == [0, 1, 0]


def test_pick_tier_edges_returns_low_below_high() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    proba = np.asarray([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
    edges = pick_tier_edges(labels, proba)
    assert edges.low < edges.high
