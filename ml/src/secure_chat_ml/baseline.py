"""Train and evaluate the TF-IDF + URL-feature + Logistic Regression baseline.

This module holds testable, importable logic; `scripts/train_baseline.py` is
the thin CLI wrapper that points it at the real downloaded (or rewritten)
corpora. Keeping the logic here lets unit tests exercise it against tiny
synthetic data without requiring the multi-gigabyte raw datasets.

Training protocol:
1. Fit the feature union and classifier on TRAIN only (never val/test).
2. Search a small C grid and a decision-threshold grid on VALIDATION only.
3. Selection rule: maximize scam recall subject to legitimate recall
   staying at or above a floor (default 0.85), so most ordinary messages
   are not warned. A few false alarms are acceptable; flooding ham is not.
   If that floor is infeasible, pick the threshold with the best scam F1
   and record that fallback.
4. Freeze C and the threshold, then score TEST once for reported metrics.
"""

# Import dataclass to bundle evaluation and tuning results without a bare dict.
from dataclasses import dataclass, field

# Import Path for typed, portable filesystem locations.
from pathlib import Path

# Import Any for the loosely typed sklearn classification_report dictionary.
from typing import Any

# Import matplotlib's non-interactive backend before pyplot for headless CI/servers.
import matplotlib

matplotlib.use("Agg")

# Import pyplot only for saving the confusion-matrix figure to disk.
import matplotlib.pyplot as plt

# Import numpy to apply a probability threshold without pandas alignment issues.
import numpy as np

# Import pandas for loading and concatenating the normalized corpus CSVs.
import pandas as pd

# Import seaborn for a readable annotated confusion-matrix heatmap.
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Import scikit-learn's vectorizer, classifier, pipeline, and evaluation metrics.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

# Import the local URL-feature branch (lexical/structural only; no network I/O).
from secure_chat_ml.url_features import build_url_feature_pipeline

# Name the two classes exactly as data/label-schema.yaml defines them.
LEGITIMATE_LABEL = 0
SCAM_LABEL = 1
CLASS_NAMES = ("legitimate", "scam")

# Reject rows whose text is empty after whitespace normalization.
_MIN_TEXT_LENGTH = 1

# Default C grid searched on validation; skipped when --no-tune-threshold is set.
DEFAULT_C_GRID: tuple[float, ...] = (0.25, 1.0, 4.0)

# Default decision thresholds on predict_proba[:, scam], inclusive of 0.30 and 0.70.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(i / 100 for i in range(30, 71, 5))

# Legitimate-recall floor: warn on at most ~15% of real ham (default 0.85).
DEFAULT_LEGIT_RECALL_FLOOR = 0.85

# Human-readable copy of the default selection rule, shared with DistilBERT reports.
DEFAULT_SELECTION_RULE = (
    f"maximize scam recall subject to legitimate recall >= {DEFAULT_LEGIT_RECALL_FLOOR:.2f}"
)


# Bundle the numbers a portfolio reviewer actually needs to see, together.
@dataclass
class BaselineEvaluation:
    """Represent one trained baseline's evaluation results."""

    # Record how many rows trained this run, for reproducibility notes.
    train_rows: int
    # Record how many rows were scored (test, val, or an external set).
    test_rows: int
    # Record the full precision/recall/F1/support breakdown, per class and averaged.
    classification_report: dict[str, Any]
    # Record the raw 2x2 confusion matrix as nested lists (JSON-serializable).
    confusion_matrix: list[list[int]]
    # Record per-source row counts so class imbalance across corpora stays visible.
    source_counts: dict[str, int] = field(default_factory=dict)
    # Record the decision threshold applied to predict_proba[:, scam].
    threshold: float = 0.5
    # Record the LogisticRegression C used for this evaluation.
    C: float = 1.0
    # Record validation-split size when the caller scored a three-way split.
    val_rows: int = 0


# Bundle the frozen validation choices so TEST scoring and chat-eval can reuse them.
@dataclass
class ThresholdTuningResult:
    """Represent the C and threshold frozen on the validation split only."""

    # Record the selected inverse-regularization strength.
    C: float
    # Record the selected probability threshold for the scam class.
    threshold: float
    # Record why this pair was chosen (floor feasible vs F1 fallback).
    selection_reason: str
    # Record the human-readable selection rule applied.
    selection_rule: str
    # Record the legitimate-recall floor used during selection.
    legit_recall_floor: float
    # Record whether any grid point met the legitimate-recall floor.
    floor_feasible: bool
    # Record the validation classification_report at the chosen operating point.
    classification_report: dict[str, Any]
    # Record the validation confusion matrix at the chosen operating point.
    confusion_matrix: list[list[int]]
    # Record how many validation rows were used (never the locked chat eval set).
    val_rows: int
    # Record the C values that were searched.
    grid_C: list[float] = field(default_factory=list)
    # Record the thresholds that were searched.
    grid_thresholds: list[float] = field(default_factory=list)


# Load and concatenate every normalized corpus CSV into one labeled dataset.
def load_processed_corpora(processed_dir: Path) -> pd.DataFrame:
    """Return one deduplicated DataFrame combining every processed corpus CSV."""

    # Find every corpus file the download/normalize (or rewrite) scripts have produced.
    csv_paths = sorted(processed_dir.glob("*.csv"))
    # Fail loudly rather than silently training on zero rows.
    if not csv_paths:
        raise FileNotFoundError(
            f"No processed corpus CSVs found under {processed_dir}. "
            "Run the ml/scripts/download_*.py scripts first, then "
            "scripts/rewrite_chat_register_llm.py for data/processed_chat_llm "
            "or scripts/rewrite_chat_register.py for data/processed_chat."
        )
    # Refuse to load the locked chat-style eval set even if a path is mis-pointed.
    if "chat_eval" in processed_dir.parts:
        # Honor evaluation_policy.chat_style_eval_training_allowed: false.
        raise ValueError(
            f"Refusing to load {processed_dir} for training: "
            "chat_style_eval_training_allowed is false."
        )
    # Read and concatenate every corpus, keeping the shared label-schema columns.
    frames = [pd.read_csv(path) for path in csv_paths]
    combined = pd.concat(frames, ignore_index=True)
    # Drop rows with missing or empty text; they carry no signal for TF-IDF.
    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined = combined[combined["text"].str.len() >= _MIN_TEXT_LENGTH]
    # Drop exact-duplicate text across corpora so one message cannot dominate the split.
    combined = combined.drop_duplicates(subset="text", keep="first")
    # Reset the index so downstream positional operations behave predictably.
    return combined.reset_index(drop=True)


# Infer rewrite_method from a processed-corpora directory name (never from chat_eval).
def infer_rewrite_method(processed_dir: Path) -> str:
    """Return llm_intent_v1, rule_based_v1, none, or unknown from the directory name."""

    # Use the final path component so relative and absolute paths both work.
    name = Path(processed_dir).name
    # LLM intent-preserving DMs live under data/processed_chat_llm.
    if name == "processed_chat_llm":
        # Stamp reports with the documented LLM rewrite identifier.
        return "llm_intent_v1"
    # Deterministic strip/shorten DMs live under data/processed_chat.
    if name == "processed_chat":
        # Stamp reports with the documented rule-based rewrite identifier.
        return "rule_based_v1"
    # Original email/SMS corpora live under data/processed.
    if name == "processed":
        # No rewrite was applied.
        return "none"
    # Unknown directory names stay explicit so a typo cannot look like a known method.
    return "unknown"


# Split the combined corpus into stratified train, validation, and test partitions.
def stratified_split(
    frame: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.20,
    test_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with class balance preserved in all splits.

    Implementation: nested sklearn train_test_split — hold out test_size first,
    then take val_size / (train_size + val_size) of the remainder as validation.
    """

    # Require the three fractions to form a complete partition.
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        # Fail rather than silently renormalizing a reviewer's typo.
        raise ValueError(
            f"train_size ({train_size}) + val_size ({val_size}) + "
            f"test_size ({test_size}) must sum to 1.0."
        )
    # Hold out the test split first so it is never used for fitting or threshold search.
    rest_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame["label"],
    )
    # Convert the requested val fraction of the full set into a fraction of the remainder.
    val_fraction_of_rest = val_size / (train_size + val_size)
    # Split the remainder into train and validation, still stratified on the binary label.
    train_df, val_df = train_test_split(
        rest_df,
        test_size=val_fraction_of_rest,
        random_state=random_state,
        stratify=rest_df["label"],
    )
    # Return fresh, independently indexed copies for downstream use.
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# Build the TF-IDF + local-URL FeatureUnion used by the classifier head.
def build_feature_union(max_features: int = 50_000) -> FeatureUnion:
    """Return an unfitted FeatureUnion of TF-IDF text features and URL features."""

    # Combine sparse TF-IDF with the scaled, sparsified URL block.
    return FeatureUnion(
        transformer_list=[
            # Vectorize unigrams and bigrams; sublinear TF dampens repeated scam boilerplate.
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            # Add on-device URL lexical/structural features; zeros when a message has no URL.
            (
                "url",
                build_url_feature_pipeline(),
            ),
        ]
    )


# Build the LogisticRegression head with a chosen C.
def build_classifier(C: float = 1.0, random_state: int = 42) -> LogisticRegression:
    """Return an unfitted class-weighted logistic classifier."""

    # Use class_weight="balanced" because scam/legitimate rows are not 50/50.
    return LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )


# Build the untrained TF-IDF + URL features + Logistic Regression pipeline.
def build_pipeline(max_features: int = 50_000, C: float = 1.0) -> Pipeline:
    """Return an unfitted scikit-learn Pipeline implementing the A5 baseline.

    Per decision A5, only this pipeline's LogisticRegression head is exported
    to ONNX in a later slice; TF-IDF vectorization itself is reimplemented in
    TypeScript from the fitted vocabulary_/idf_, because ai.onnx.ml's
    Tokenizer/StringNormalizer operators are unsupported in ONNX Runtime Web.
    URL features are computed locally in the same spirit (no live reputation).
    """

    # Stack the feature union and the logistic head into one fit/predict object.
    return Pipeline(
        steps=[
            # Transform raw message text into TF-IDF plus scaled URL features.
            ("features", build_feature_union(max_features=max_features)),
            # Classify with a class-weighted logistic head at the chosen C.
            ("classifier", build_classifier(C=C)),
        ]
    )


# Turn scam probabilities into hard labels using a frozen threshold.
def predict_with_threshold(
    pipeline: Pipeline,
    texts: pd.Series,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return binary predictions using predict_proba[:, scam] >= threshold."""

    # Score the scam class probability for every row.
    probabilities = pipeline.predict_proba(texts)[:, SCAM_LABEL]
    # Apply the caller-provided threshold instead of sklearn's implicit 0.5.
    return (probabilities >= threshold).astype(int)


# Build a BaselineEvaluation from already-computed predictions (no fitting).
def evaluation_from_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    train_rows: int,
    source_frame: pd.DataFrame | None = None,
    threshold: float = 0.5,
    C: float = 1.0,
    val_rows: int = 0,
) -> BaselineEvaluation:
    """Return precision/recall/F1/confusion matrix for a labeled prediction vector."""

    # Build the full per-class and averaged precision/recall/F1/support breakdown.
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    # Build the 2x2 confusion matrix with a fixed label order for reproducible axes.
    matrix = confusion_matrix(y_true, y_pred, labels=[LEGITIMATE_LABEL, SCAM_LABEL])
    # Count rows per source corpus when the caller provided the original frame.
    source_counts = (
        source_frame["source"].value_counts().to_dict()
        if source_frame is not None and "source" in source_frame.columns
        else {}
    )
    # Bundle the metrics for JSON reports and tests.
    return BaselineEvaluation(
        train_rows=train_rows,
        test_rows=len(y_true),
        classification_report=report,
        confusion_matrix=matrix.tolist(),
        source_counts=source_counts,
        threshold=threshold,
        C=C,
        val_rows=val_rows,
    )


# Fit the pipeline on TRAIN and score another split with an optional threshold.
def evaluate(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
) -> BaselineEvaluation:
    """Fit on the training split and evaluate precision/recall/F1/confusion matrix.

    Predictions use predict_proba[:, scam] >= threshold (default 0.5), not only
    sklearn's implicit 0.5 cut from Pipeline.predict.
    """

    # Fit the vectorizer, URL scaler, and classifier together on TRAIN only.
    pipeline.fit(train_df["text"], train_df["label"])
    # Predict on the held-out split using the caller-provided decision threshold.
    predictions = predict_with_threshold(pipeline, test_df["text"], threshold=threshold)
    # Read C from the fitted logistic head when present, else default to 1.0.
    classifier = pipeline.named_steps.get("classifier")
    # Use the fitted C so reports stay consistent with the actual model.
    fitted_C = float(getattr(classifier, "C", 1.0))
    # Build the metrics bundle from the thresholded predictions.
    return evaluation_from_predictions(
        test_df["label"],
        predictions,
        train_rows=len(train_df),
        source_frame=test_df,
        threshold=threshold,
        C=fitted_C,
    )


# Score a frozen probability vector at every documented decision threshold.
def score_threshold_grid(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    *,
    C: float = 0.0,
) -> list[dict[str, Any]]:
    """Return one metrics dict per threshold without fitting a classifier.

    DistilBERT and the TF-IDF baseline share this helper so the VAL grid and
    the reported confusion-matrix orientation stay identical.
    """

    # Accumulate every (threshold, precision/recall/F1) point for later selection.
    candidates: list[dict[str, Any]] = []
    # Cast labels and probabilities so list callers and numpy callers both work.
    y_true_array = np.asarray(y_true)
    # Cast probabilities to float so the threshold comparison is vectorized.
    y_proba_array = np.asarray(y_proba, dtype=float)
    # Sweep the documented grid; probabilities are already computed.
    for threshold in threshold_grid:
        # Convert scam probabilities into hard labels at this operating point.
        y_pred = (y_proba_array >= threshold).astype(int)
        # Compute the full report so precision/recall/F1 are all available.
        report = classification_report(
            y_true_array,
            y_pred,
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        )
        # Store the point for the later selection pass.
        candidates.append(
            {
                "C": float(C),
                "threshold": float(threshold),
                "report": report,
                "matrix": confusion_matrix(
                    y_true_array, y_pred, labels=[LEGITIMATE_LABEL, SCAM_LABEL]
                ).tolist(),
                "legit_precision": float(report["legitimate"]["precision"]),
                "legit_recall": float(report["legitimate"]["recall"]),
                "scam_recall": float(report["scam"]["recall"]),
                "scam_f1": float(report["scam"]["f1-score"]),
            }
        )
    # Return the full grid so callers can apply the shared selection rule.
    return candidates


# Apply the documented VAL selection rule to an already-scored threshold grid.
def pick_operating_point(
    candidates: list[dict[str, Any]],
    legit_recall_floor: float = DEFAULT_LEGIT_RECALL_FLOOR,
) -> tuple[dict[str, Any], str, bool]:
    """Return (chosen_row, selection_reason, floor_feasible) from scored candidates.

    Maximizes scam recall among points whose legitimate recall is at least
    `legit_recall_floor`. If none qualify, picks the best scam F1 instead.
    """

    # Fail loudly on an empty grid rather than inventing a threshold.
    if not candidates:
        # A missing grid is a caller bug, not a feasible fallback.
        raise ValueError("Cannot pick an operating point from an empty candidate list.")
    # Restrict to operating points that leave most real ham unwarned.
    feasible = [row for row in candidates if row["legit_recall"] >= legit_recall_floor]
    # Prefer the feasible set; fall back to the full grid when the floor is infeasible.
    if feasible:
        # Maximize scam recall; among ties keep more ham, then the higher threshold.
        chosen = max(
            feasible,
            key=lambda row: (row["scam_recall"], row["legit_recall"], row["threshold"]),
        )
        # Record that the ham-recall floor was met.
        return chosen, "max_scam_recall_subject_to_legit_recall_floor", True
    # No point met the floor; pick the best scam F1 on the full grid as specified.
    chosen = max(candidates, key=lambda row: (row["scam_f1"], row["scam_recall"]))
    # Record the documented fallback so the README can say so honestly.
    return chosen, "legit_recall_floor_infeasible_max_scam_f1", False


# Search C and the decision threshold on VALIDATION only, never on test or chat eval.
def tune_on_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    max_features: int = 50_000,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    legit_recall_floor: float = DEFAULT_LEGIT_RECALL_FLOOR,
    random_state: int = 42,
) -> ThresholdTuningResult:
    """Return the frozen C and threshold chosen on the validation split.

    Features are fit on TRAIN only. For each C a logistic head is fit on the
    already-transformed TRAIN matrix, then every threshold is scored on VAL.
    Selection maximizes scam recall among points whose legitimate recall is
    at least `legit_recall_floor` (at most ~15% of ham warned at the default).
    If none qualify, the point with the best scam F1 is chosen and
    `selection_reason` records the fallback.
    """

    # Fit the FeatureUnion on TRAIN text only so validation cannot leak into IDF or scaling.
    features = build_feature_union(max_features=max_features)
    # Transform TRAIN into the sparse TF-IDF + URL block.
    X_train = features.fit_transform(train_df["text"])
    # Transform VAL with TRAIN-fitted IDF and URL scaler (no refit).
    X_val = features.transform(val_df["text"])
    # Read the binary labels as numpy arrays for sklearn.
    y_train = train_df["label"].to_numpy()
    # Read validation labels once for every grid point.
    y_val = val_df["label"].to_numpy()
    # Accumulate every (C, threshold, metrics) point for selection.
    candidates: list[dict[str, Any]] = []
    # Fit one logistic head per C; thresholds are applied to predict_proba without refitting.
    for C in C_grid:
        # Build a fresh class-weighted logistic head at this C.
        classifier = build_classifier(C=C, random_state=random_state)
        # Announce the fit so a long C-grid run is auditable from the terminal.
        print(f"tune_on_validation: fitting LogisticRegression(C={C}) on TRAIN...")
        # Fit on the TRAIN feature matrix only.
        classifier.fit(X_train, y_train)
        # Score scam probabilities on VALIDATION only.
        val_proba = classifier.predict_proba(X_val)[:, SCAM_LABEL]
        # Reuse the shared threshold scorer so DistilBERT and TF-IDF share one grid.
        candidates.extend(
            score_threshold_grid(y_val, val_proba, threshold_grid, C=float(C))
        )
    # Document the selection rule in the result so reports stay auditable.
    selection_rule = (
        f"maximize scam recall subject to legitimate recall >= {legit_recall_floor:.2f}"
    )
    # Apply the same floor-then-F1-fallback rule DistilBERT uses on VAL.
    chosen, selection_reason, floor_feasible = pick_operating_point(
        candidates, legit_recall_floor=legit_recall_floor
    )
    # Bundle the frozen choices and the validation metrics at that operating point.
    return ThresholdTuningResult(
        C=float(chosen["C"]),
        threshold=float(chosen["threshold"]),
        selection_reason=selection_reason,
        selection_rule=selection_rule,
        legit_recall_floor=float(legit_recall_floor),
        floor_feasible=floor_feasible,
        classification_report=chosen["report"],
        confusion_matrix=chosen["matrix"],
        val_rows=len(val_df),
        grid_C=[float(value) for value in C_grid],
        grid_thresholds=[float(value) for value in threshold_grid],
    )


# Load the hand-curated, chat-style, evaluation-only dataset.
def load_chat_style_eval_set(path: Path) -> pd.DataFrame:
    """Return the chat-style eval set, failing loudly if it has not been built yet.

    Per data/label-schema.yaml's evaluation_policy, callers must never pass
    the result of this function to Pipeline.fit(...) — only .predict(...).
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No chat-style eval set found at {path}. "
            "Run scripts/build_chat_style_eval_set.py first."
        )
    frame = pd.read_csv(path)
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    return frame.reset_index(drop=True)


# Evaluate an already-fitted pipeline against external (non-training) data.
def evaluate_external(
    pipeline: Pipeline,
    external_df: pd.DataFrame,
    threshold: float = 0.5,
) -> BaselineEvaluation:
    """Score a fitted pipeline's predictions on data it was never trained on.

    Unlike evaluate(), this never calls pipeline.fit(...): the pipeline
    passed in must already be fitted on in-domain training data. This is
    the function scripts/evaluate_chat_style_eval.py uses so the hand-curated
    chat-style set is scored, never trained on. The frozen validation
    threshold is applied here; it is never retuned on the locked eval rows.
    """

    # Predict with the frozen threshold rather than sklearn's implicit 0.5.
    predictions = predict_with_threshold(pipeline, external_df["text"], threshold=threshold)
    # Read C from the fitted head when present so the JSON report is complete.
    classifier = pipeline.named_steps.get("classifier")
    # Default to 1.0 when the pipeline was built without a named classifier step.
    fitted_C = float(getattr(classifier, "C", 1.0))
    # Build the metrics bundle; train_rows stays 0 because this function never fits.
    return evaluation_from_predictions(
        external_df["label"],
        predictions,
        train_rows=0,
        source_frame=external_df,
        threshold=threshold,
        C=fitted_C,
    )


# Render the confusion matrix as an annotated heatmap saved to disk.
def save_confusion_matrix_plot(
    evaluation: BaselineEvaluation,
    output_path: Path,
    title: str = "TF-IDF + URL features + Logistic Regression (test)",
) -> None:
    """Save a labeled confusion-matrix heatmap for the README and reports."""

    # Ensure the parent directory exists before writing the figure.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Build a fresh figure sized for a readable 2x2 heatmap.
    figure, axis = plt.subplots(figsize=(5, 4))
    # Draw the confusion matrix with integer annotations and a readable colormap.
    sns.heatmap(
        evaluation.confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    # Label axes so the plot is self-explanatory outside this notebook/script.
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    # Use the caller-provided title so DistilBERT reports are not mislabeled TF-IDF.
    axis.set_title(title)
    # Save without extra whitespace so the image embeds cleanly in the README.
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    # Release the figure so repeated calls (e.g. in tests) do not leak memory.
    plt.close(figure)
