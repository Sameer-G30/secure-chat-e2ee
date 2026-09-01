"""Two-threshold cascade and stacked/soft-vote ensembles over existing checkpoints.

All three base models (TF-IDF, Word BiLSTM, DistilBERT) have already seen TRAIN.
This module therefore never fits a meta-learner on TRAIN predictions. The
stacker is fit on one half of VALIDATION; the cascade band edges and the
stacker's operating point are chosen on the other half. The locked
chat_style_eval_v1.csv file is never used here.

A cascade that escalates to DistilBERT still requires shipping ~64 MiB to every
browser. The two-tier TF-IDF→BiLSTM path is recorded as a separate arm so the
deployability constraint can decide the winner after measurement.
"""

# Import dataclasses to bundle frozen cascade edges without a bare dict.
from dataclasses import dataclass

# Import numpy for probability vectors and boolean masks.
import numpy as np

# Import logistic regression only as the stacker's meta-learner (VAL-split fit).
from sklearn.linear_model import LogisticRegression

# Reuse the published class ids so reports stay comparable.
from secure_chat_ml.baseline import LEGITIMATE_LABEL, SCAM_LABEL

# Name the three base models in the order the cascade escalates.
TIER_TFIDF = "tfidf"
TIER_LSTM = "lstm"
TIER_DISTILBERT = "distilbert"


# Frozen two-threshold edges for one cascade tier.
@dataclass(frozen=True)
class TierEdges:
    """Represent the low (confident ham) and high (confident scam) cuts for one tier."""

    # P(scam) at or below this value is accepted as legitimate without escalating.
    low: float
    # P(scam) at or above this value is accepted as scam without escalating.
    high: float


# Frozen cascade configuration chosen on the VAL selection split only.
@dataclass(frozen=True)
class CascadeConfig:
    """Represent the two-threshold cascade over TF-IDF → BiLSTM → DistilBERT."""

    # First-tier edges (cheap TF-IDF).
    tfidf: TierEdges
    # Second-tier edges (Word BiLSTM).
    lstm: TierEdges
    # Final DistilBERT cut for the remaining ambiguous band (single threshold).
    distilbert_threshold: float
    # Record that these edges were not taken from chat_style_eval_v1.
    tuned_on: str = "validation_selection_split"


# Apply one tier's two thresholds: confident ham, confident scam, or "escalate".
def classify_tier(
    proba: np.ndarray,
    edges: TierEdges,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (decision, escalate_mask).

    decision is 0, 1, or -1 (undecided / escalate). escalate_mask is True
    where the probability sits strictly between the two edges.
    """

    # Start every row as "escalate" so the band is the default, not an afterthought.
    decision = np.full(proba.shape, -1, dtype=np.int64)
    # Confident legitimate: at or below the low edge.
    decision[proba <= edges.low] = LEGITIMATE_LABEL
    # Confident scam: at or above the high edge.
    decision[proba >= edges.high] = SCAM_LABEL
    # Escalate when the row is still undecided.
    escalate = decision == -1
    return decision, escalate


# Run TF-IDF → BiLSTM → DistilBERT, stopping at the first confident tier.
def apply_cascade(
    p_tfidf: np.ndarray,
    p_lstm: np.ndarray,
    p_distilbert: np.ndarray,
    config: CascadeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, tier_used) for every row.

    tier_used is 0 (TF-IDF), 1 (LSTM), or 2 (DistilBERT). Rows that DistilBERT
    still scores use config.distilbert_threshold as a single cut.
    """

    # Align shapes so a caller cannot silently zip mismatched vectors.
    if not (p_tfidf.shape == p_lstm.shape == p_distilbert.shape):
        raise ValueError("cascade probability vectors must have the same shape")
    labels = np.full(p_tfidf.shape, LEGITIMATE_LABEL, dtype=np.int64)
    tier_used = np.zeros(p_tfidf.shape, dtype=np.int64)
    tfidf_decision, escalate_from_tfidf = classify_tier(p_tfidf, config.tfidf)
    settled = ~escalate_from_tfidf
    labels[settled] = tfidf_decision[settled]
    tier_used[settled] = 0
    if not np.any(escalate_from_tfidf):
        return labels, tier_used
    lstm_decision, escalate_from_lstm = classify_tier(p_lstm, config.lstm)
    lstm_settled = escalate_from_tfidf & ~escalate_from_lstm
    labels[lstm_settled] = lstm_decision[lstm_settled]
    tier_used[lstm_settled] = 1
    distilbert_rows = escalate_from_tfidf & escalate_from_lstm
    labels[distilbert_rows] = (
        p_distilbert[distilbert_rows] >= config.distilbert_threshold
    ).astype(np.int64)
    tier_used[distilbert_rows] = 2
    return labels, tier_used


# Apply two thresholds, then a single fallback cut to the remaining band.
def apply_two_threshold_with_fallback(
    proba: np.ndarray,
    edges: TierEdges,
    fallback_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, used_fallback) for one probability vector.

    Confident rows take the two-threshold decision. The band uses
    `fallback_threshold` as a single cut. When that cut sits between the
    two edges this is identical to a one-threshold classifier; it exists so
    a missing LSTM/DistilBERT checkpoint still produces a measurable arm
    instead of skipping the cascade code path.
    """

    # Settle the confident tails first.
    decision, escalate = classify_tier(proba, edges)
    # Copy so the escalate rows can be overwritten without mutating decision.
    labels = decision.copy()
    # Score the ambiguous band with the VAL-frozen single threshold.
    labels[escalate] = (proba[escalate] >= fallback_threshold).astype(np.int64)
    # Record which rows used the fallback cut (1) vs the two-threshold tails (0).
    used_fallback = escalate.astype(np.int64)
    return labels, used_fallback


# Two-threshold policy that refuses to warn on the ambiguous band.
def apply_two_threshold_band_as_legitimate(
    proba: np.ndarray,
    edges: TierEdges,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, in_band) treating the escalate band as legitimate.

    This is the TF-IDF-only false-alarm attack: warn only when P(scam) is
    at or above the high edge. The band is not a second model; it is an
    explicit "do not banner" decision so a missing BiLSTM checkpoint still
    yields a number on chat_style_eval_v2.
    """

    # Settle confident ham/scam; the rest of the probability range abstains.
    decision, escalate = classify_tier(proba, edges)
    # Copy so abstain rows can be filled without mutating the tier output.
    labels = decision.copy()
    # Treat "not confident this is a scam" as legitimate (no banner).
    labels[escalate] = LEGITIMATE_LABEL
    # Record which rows sat in the band for the coverage report.
    in_band = escalate.astype(np.int64)
    return labels, in_band


# Two-tier cascade that never loads DistilBERT (the shippable browser path).
def apply_two_tier_cascade(
    p_tfidf: np.ndarray,
    p_lstm: np.ndarray,
    tfidf_edges: TierEdges,
    lstm_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, tier_used) using only TF-IDF and BiLSTM.

    Ambiguous TF-IDF rows fall through to a single LSTM threshold. This is the
    arm that can ship in the browser without the 64 MiB DistilBERT graph.
    """

    if p_tfidf.shape != p_lstm.shape:
        raise ValueError("two-tier probability vectors must have the same shape")
    labels = np.full(p_tfidf.shape, LEGITIMATE_LABEL, dtype=np.int64)
    tier_used = np.zeros(p_tfidf.shape, dtype=np.int64)
    tfidf_decision, escalate = classify_tier(p_tfidf, tfidf_edges)
    settled = ~escalate
    labels[settled] = tfidf_decision[settled]
    labels[escalate] = (p_lstm[escalate] >= lstm_threshold).astype(np.int64)
    tier_used[escalate] = 1
    return labels, tier_used


# Unweighted mean of the three P(scam) outputs (no extra fitted parameters).
def soft_vote(
    p_tfidf: np.ndarray,
    p_lstm: np.ndarray,
    p_distilbert: np.ndarray,
) -> np.ndarray:
    """Return the arithmetic mean of three probability vectors."""

    if not (p_tfidf.shape == p_lstm.shape == p_distilbert.shape):
        raise ValueError("soft-vote probability vectors must have the same shape")
    return (p_tfidf + p_lstm + p_distilbert) / 3.0


# Fit a logistic stacker on VAL-A predictions only (never TRAIN, never chat eval).
def fit_stacker(
    p_tfidf: np.ndarray,
    p_lstm: np.ndarray,
    p_distilbert: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """Return a 3-feature logistic regression fit on the stacking split."""

    features = np.column_stack([p_tfidf, p_lstm, p_distilbert])
    # A linear stacker on three probabilities; C=1.0 is not searched on TEST.
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(features, labels)
    return model


# Score a fitted stacker; the caller still applies a VAL-B-frozen threshold.
def stacker_proba(
    model: LogisticRegression,
    p_tfidf: np.ndarray,
    p_lstm: np.ndarray,
    p_distilbert: np.ndarray,
) -> np.ndarray:
    """Return P(scam) from the stacked logistic meta-learner."""

    features = np.column_stack([p_tfidf, p_lstm, p_distilbert])
    return model.predict_proba(features)[:, SCAM_LABEL]


# Search a two-threshold grid for one tier on the VAL selection split only.
def pick_tier_edges(
    labels: np.ndarray,
    proba: np.ndarray,
    *,
    low_grid: tuple[float, ...] = (0.10, 0.20, 0.30),
    high_grid: tuple[float, ...] = (0.70, 0.80, 0.90),
    min_coverage: float = 0.20,
) -> TierEdges:
    """Return low/high edges that settle the most rows without harming accuracy.

    Coverage is the fraction of rows that do not escalate. Among pairs whose
    settled accuracy is at least as good as always-predicting-the-majority,
    pick the pair with the highest coverage so the expensive next tier is used
    as little as possible. This search never sees chat_style_eval_v1.
    """

    majority = int(np.round(np.mean(labels)))
    best: tuple[float, float, float] | None = None
    for low in low_grid:
        for high in high_grid:
            if low >= high:
                continue
            decision, escalate = classify_tier(proba, TierEdges(low=low, high=high))
            settled = ~escalate
            coverage = float(np.mean(settled))
            if coverage < min_coverage or not np.any(settled):
                continue
            accuracy = float(np.mean(decision[settled] == labels[settled]))
            majority_accuracy = float(np.mean(labels[settled] == majority))
            if accuracy < majority_accuracy:
                continue
            score = coverage
            if best is None or score > best[0]:
                best = (score, low, high)
    if best is None:
        # Fall back to a conservative band that still escalates the middle.
        return TierEdges(low=0.20, high=0.80)
    return TierEdges(low=best[1], high=best[2])


# Split VALIDATION into a stacking half and a selection half without touching TRAIN.
def split_validation_for_meta(
    n: int,
    *,
    random_state: int = 42,
    stack_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (stack_idx, select_idx) as a disjoint partition of range(n)."""

    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)
    cut = int(round(n * stack_fraction))
    return order[:cut], order[cut:]
