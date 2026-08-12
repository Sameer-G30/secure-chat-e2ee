"""Train and evaluate the TF-IDF + Logistic Regression scam-detection baseline.

This module holds testable, importable logic; `scripts/train_baseline.py` is
the thin CLI wrapper that points it at the real downloaded corpora. Keeping
the logic here lets unit tests exercise it against tiny synthetic data
without requiring the multi-gigabyte raw datasets to be present.
"""

# Import dataclass to bundle evaluation results without a bare dict.
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

# Import pandas for loading and concatenating the normalized corpus CSVs.
import pandas as pd

# Import seaborn for a readable annotated confusion-matrix heatmap.
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Import scikit-learn's vectorizer, classifier, pipeline, and evaluation metrics.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Name the two classes exactly as data/label-schema.yaml defines them.
LEGITIMATE_LABEL = 0
SCAM_LABEL = 1
CLASS_NAMES = ("legitimate", "scam")

# Reject rows whose text is empty after whitespace normalization.
_MIN_TEXT_LENGTH = 1


# Bundle the numbers a portfolio reviewer actually needs to see, together.
@dataclass
class BaselineEvaluation:
    """Represent one trained baseline's evaluation results."""

    # Record how many rows trained and tested this run, for reproducibility notes.
    train_rows: int
    test_rows: int
    # Record the full precision/recall/F1/support breakdown, per class and averaged.
    classification_report: dict[str, Any]
    # Record the raw 2x2 confusion matrix as nested lists (JSON-serializable).
    confusion_matrix: list[list[int]]
    # Record per-source row counts so class imbalance across corpora stays visible.
    source_counts: dict[str, int] = field(default_factory=dict)


# Load and concatenate every normalized corpus CSV into one labeled dataset.
def load_processed_corpora(processed_dir: Path) -> pd.DataFrame:
    """Return one deduplicated DataFrame combining every processed corpus CSV."""

    # Find every corpus file the download/normalize scripts have produced so far.
    csv_paths = sorted(processed_dir.glob("*.csv"))
    # Fail loudly rather than silently training on zero rows.
    if not csv_paths:
        raise FileNotFoundError(
            f"No processed corpus CSVs found under {processed_dir}. "
            "Run the ml/scripts/download_*.py scripts first."
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


# Split the combined corpus into stratified train and test partitions.
def stratified_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with class balance preserved in both splits."""

    # Stratify on the binary label so the imbalanced classes split proportionally.
    train_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame["label"],
    )
    # Return fresh, independently indexed copies for downstream use.
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Build the untrained TF-IDF + Logistic Regression pipeline.
def build_pipeline(max_features: int = 50_000) -> Pipeline:
    """Return an unfitted scikit-learn Pipeline implementing the A5 baseline.

    Per decision A5, only this pipeline's LogisticRegression head is exported
    to ONNX in a later slice; TF-IDF vectorization itself is reimplemented in
    TypeScript from the fitted vocabulary_/idf_, because ai.onnx.ml's
    Tokenizer/StringNormalizer operators are unsupported in ONNX Runtime Web.
    """

    return Pipeline(
        steps=[
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
            # Use class_weight="balanced" because scam/legitimate rows are not 50/50.
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# Fit the pipeline and compute the metrics the spec requires (never accuracy alone).
def evaluate(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> BaselineEvaluation:
    """Fit on the training split and evaluate precision/recall/F1/confusion matrix."""

    # Fit the vectorizer and classifier together on the training partition only.
    pipeline.fit(train_df["text"], train_df["label"])
    # Predict on the held-out test partition the model never saw during fitting.
    predictions = pipeline.predict(test_df["text"])
    # Build the full per-class and averaged precision/recall/F1/support breakdown.
    report = classification_report(
        test_df["label"],
        predictions,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    # Build the 2x2 confusion matrix with a fixed label order for reproducible axes.
    matrix = confusion_matrix(test_df["label"], predictions, labels=[LEGITIMATE_LABEL, SCAM_LABEL])
    # Count rows per source corpus so the README can discuss class imbalance honestly.
    source_counts = (
        test_df["source"].value_counts().to_dict() if "source" in test_df.columns else {}
    )
    return BaselineEvaluation(
        train_rows=len(train_df),
        test_rows=len(test_df),
        classification_report=report,
        confusion_matrix=matrix.tolist(),
        source_counts=source_counts,
    )


# Render the confusion matrix as an annotated heatmap saved to disk.
def save_confusion_matrix_plot(evaluation: BaselineEvaluation, output_path: Path) -> None:
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
    axis.set_title("TF-IDF + Logistic Regression baseline")
    # Save without extra whitespace so the image embeds cleanly in the README.
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    # Release the figure so repeated calls (e.g. in tests) do not leak memory.
    plt.close(figure)
