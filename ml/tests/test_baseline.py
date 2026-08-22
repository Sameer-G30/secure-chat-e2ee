"""Exercise the baseline pipeline against tiny synthetic data, not real corpora.

These tests never require the multi-gigabyte downloaded datasets, so they run
safely in CI without network access or the ml/data/raw fixtures.
"""

# Import Path for the temporary synthetic-corpus fixture directory.
from pathlib import Path

# Import pandas to build the synthetic corpus fixture DataFrame.
import pandas as pd

# Import pytest's tmp_path fixture typing implicitly through the function signature.
import pytest

from secure_chat_ml.baseline import (
    CLASS_NAMES,
    DEFAULT_C_GRID,
    DEFAULT_THRESHOLD_GRID,
    EXPANDED_THRESHOLD_GRID,
    WIDENED_C_GRID,
    BaselineHyperparameters,
    build_pipeline,
    evaluate,
    hyperparameters_from_mapping,
    hyperparameters_to_dict,
    infer_rewrite_method,
    load_processed_corpora,
    pick_operating_point,
    predict_with_threshold,
    save_confusion_matrix_plot,
    score_threshold_grid,
    stratified_split,
    tune_on_validation,
)

# Build enough repeated-pattern rows that TF-IDF has real signal to separate classes.
_LEGITIMATE_TEXTS = [
    "let's grab lunch tomorrow at noon",
    "can you send me the meeting notes",
    "happy birthday, hope you have a great day",
    "the report is attached, let me know if edits are needed",
    "see you at the gym later this evening",
    "thanks for helping me move last weekend",
    "the weather looks nice for our hike on saturday",
    "here is the recipe you asked about",
    "let's reschedule our call to friday afternoon",
    "great job on the presentation today",
] * 6
_SCAM_TEXTS = [
    "urgent: your account will be suspended, verify your password now",
    "you have won a prize, click this link to claim your reward",
    "congratulations, you are eligible for a free gift card, act now",
    "your bank account has been locked, confirm your login immediately",
    "limited time offer, wire money now to secure this deal",
    "verify your identity within 24 hours or lose access to your account",
    "you have an unclaimed refund, provide your card details to receive it",
    "final notice: update your payment information to avoid suspension",
    "click here immediately to unlock your frozen account",
    "act now, this exclusive investment opportunity expires today",
] * 6


# Write a tiny synthetic corpus CSV shaped like the real label-schema output.
@pytest.fixture
def synthetic_processed_dir(tmp_path: Path) -> Path:
    """Return a directory containing one small, schema-shaped corpus CSV."""

    rows = []
    for index, text in enumerate(_LEGITIMATE_TEXTS):
        rows.append(
            {
                "message_id": f"synthetic-legit-{index:04d}",
                "text": text,
                "label": 0,
                "original_label": "ham",
                "source": "synthetic_test_corpus",
                "split": "unassigned",
            }
        )
    for index, text in enumerate(_SCAM_TEXTS):
        rows.append(
            {
                "message_id": f"synthetic-scam-{index:04d}",
                "text": text,
                "label": 1,
                "original_label": "scam",
                "source": "synthetic_test_corpus",
                "split": "unassigned",
            }
        )
    frame = pd.DataFrame(rows)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    frame.to_csv(processed_dir / "synthetic.csv", index=False)
    return processed_dir


# Confirm loading raises a clear error rather than silently training on nothing.
def test_load_processed_corpora_requires_at_least_one_csv(tmp_path: Path) -> None:
    """Assert an empty processed directory fails loudly."""

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_processed_corpora(empty_dir)


# Confirm a mis-pointed chat_eval directory is refused as a training source.
def test_load_processed_corpora_refuses_chat_eval_directory(tmp_path: Path) -> None:
    """Assert the locked eval directory cannot be loaded for training."""

    chat_eval_dir = tmp_path / "data" / "chat_eval"
    chat_eval_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "message_id": ["chat-eval-000"],
            "text": ["hey are we still on for lunch"],
            "label": [0],
            "original_label": ["legitimate_chat"],
            "source": ["chat_style_eval_v1"],
            "split": ["eval_only"],
        }
    )
    frame.to_csv(chat_eval_dir / "chat_style_eval_v1.csv", index=False)
    with pytest.raises(ValueError, match="chat_style_eval_training_allowed"):
        load_processed_corpora(chat_eval_dir)


# Confirm loading drops empty text and exact duplicates.
def test_load_processed_corpora_drops_empty_and_duplicate_rows(tmp_path: Path) -> None:
    """Assert dirty synthetic rows are cleaned before training ever sees them."""

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    frame = pd.DataFrame(
        {
            "message_id": ["a", "b", "c", "d"],
            "text": ["hello there", "hello there", "   ", "legit unique text"],
            "label": [0, 0, 1, 0],
            "original_label": ["ham", "ham", "spam", "ham"],
            "source": ["s", "s", "s", "s"],
            "split": ["unassigned"] * 4,
        }
    )
    frame.to_csv(processed_dir / "dirty.csv", index=False)

    combined = load_processed_corpora(processed_dir)

    # The duplicate "hello there" row and the whitespace-only row must both be gone.
    assert len(combined) == 2
    assert set(combined["text"]) == {"hello there", "legit unique text"}


# Confirm the three-way split preserves class proportions in every partition.
def test_stratified_split_preserves_class_balance(synthetic_processed_dir: Path) -> None:
    """Assert train, val, and test each contain both classes near 70/20/10."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, val_df, test_df = stratified_split(
        combined, train_size=0.70, val_size=0.20, test_size=0.10, random_state=42
    )

    # Every split must contain both classes; stratification exists precisely for this.
    assert set(train_df["label"].unique()) == {0, 1}
    assert set(val_df["label"].unique()) == {0, 1}
    assert set(test_df["label"].unique()) == {0, 1}
    # The three partitions must cover the combined corpus exactly once.
    assert len(train_df) + len(val_df) + len(test_df) == len(combined)
    # Proportions should sit near the requested 70/20/10 split (sklearn rounding).
    assert abs(len(train_df) / len(combined) - 0.70) < 0.06
    assert abs(len(val_df) / len(combined) - 0.20) < 0.06
    assert abs(len(test_df) / len(combined) - 0.10) < 0.06


# Confirm a reviewer typo in split fractions fails instead of silently renormalizing.
def test_stratified_split_rejects_sizes_that_do_not_sum_to_one(
    synthetic_processed_dir: Path,
) -> None:
    """Assert train_size + val_size + test_size must equal 1.0."""

    combined = load_processed_corpora(synthetic_processed_dir)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        stratified_split(combined, train_size=0.5, val_size=0.2, test_size=0.2)


# Confirm the full train -> evaluate path reports the metrics the spec requires.
def test_evaluate_reports_precision_recall_f1_and_confusion_matrix(
    synthetic_processed_dir: Path,
) -> None:
    """Assert evaluate() returns per-class metrics and a well-formed confusion matrix."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, _val_df, test_df = stratified_split(combined, random_state=42)
    pipeline = build_pipeline(max_features=1000)

    result = evaluate(pipeline, train_df, test_df, threshold=0.5)

    # Every class name from the shared schema must have a precision/recall/F1 entry.
    for class_name in CLASS_NAMES:
        class_metrics = result.classification_report[class_name]
        assert 0.0 <= class_metrics["precision"] <= 1.0
        assert 0.0 <= class_metrics["recall"] <= 1.0
        assert 0.0 <= class_metrics["f1-score"] <= 1.0
    # The confusion matrix must be a 2x2 grid whose total equals the test set size.
    assert len(result.confusion_matrix) == 2
    assert all(len(row) == 2 for row in result.confusion_matrix)
    assert sum(sum(row) for row in result.confusion_matrix) == result.test_rows
    # This synthetic corpus is easy to separate; the baseline should do far better than chance.
    assert result.classification_report["scam"]["f1-score"] > 0.7
    # The default threshold must be recorded on the result object.
    assert result.threshold == 0.5


# Confirm evaluate() uses the supplied threshold rather than sklearn's implicit 0.5.
def test_evaluate_honors_a_non_default_threshold(synthetic_processed_dir: Path) -> None:
    """Assert threshold=0.0 predicts every row as scam on a fitted pipeline."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, _val_df, test_df = stratified_split(combined, random_state=42)
    pipeline = build_pipeline(max_features=1000)
    # Fit once so thresholded predict can be compared without refitting.
    pipeline.fit(train_df["text"], train_df["label"])
    # A threshold of 0.0 is at or below every probability, so every row is predicted scam.
    all_scam = predict_with_threshold(pipeline, test_df["text"], threshold=0.0)
    assert set(all_scam.tolist()) == {1}
    # A threshold above 1.0 is above every probability, so every row is predicted legitimate.
    all_legit = predict_with_threshold(pipeline, test_df["text"], threshold=1.01)
    assert set(all_legit.tolist()) == {0}


# Confirm validation-only tuning returns a C and threshold from the searched grids.
def test_tune_on_validation_returns_grid_values(synthetic_processed_dir: Path) -> None:
    """Assert the selected C and threshold come from the supplied grids, not test."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, val_df, _test_df = stratified_split(combined, random_state=42)
    tuning = tune_on_validation(
        train_df,
        val_df,
        max_features=1000,
        C_grid=DEFAULT_C_GRID,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        legit_recall_floor=0.85,
        random_state=42,
    )
    # The frozen C must be one of the searched values.
    assert tuning.C in DEFAULT_C_GRID
    # The frozen threshold must be one of the searched values.
    assert tuning.threshold in DEFAULT_THRESHOLD_GRID
    # Validation row count must match the val split, never a chat-eval size.
    assert tuning.val_rows == len(val_df)
    # The recorded floor must match the ham-recall cap used for selection.
    assert tuning.legit_recall_floor == 0.85
    # The selection reason must be one of the two documented outcomes.
    assert tuning.selection_reason in {
        "max_scam_recall_subject_to_legit_recall_floor",
        "legit_recall_floor_infeasible_max_scam_f1",
    }


# Confirm an impossible ham-recall floor falls back to best scam F1.
def test_tune_on_validation_falls_back_when_recall_floor_is_infeasible(
    synthetic_processed_dir: Path,
) -> None:
    """Assert legit_recall_floor=1.01 records the documented F1 fallback."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, val_df, _test_df = stratified_split(combined, random_state=42)
    tuning = tune_on_validation(
        train_df,
        val_df,
        max_features=1000,
        C_grid=DEFAULT_C_GRID,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        legit_recall_floor=1.01,
        random_state=42,
    )
    # No classifier can achieve recall above 1.0, so the floor must be infeasible.
    assert tuning.floor_feasible is False
    # The fallback reason must name the ham-recall floor, not the old precision floor.
    assert tuning.selection_reason == "legit_recall_floor_infeasible_max_scam_f1"


# Confirm the confusion-matrix plot writes a real, non-empty image file.
def test_save_confusion_matrix_plot_writes_a_file(
    tmp_path: Path, synthetic_processed_dir: Path
) -> None:
    """Assert the saved PNG exists and has non-trivial size."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, _val_df, test_df = stratified_split(combined, random_state=42)
    pipeline = build_pipeline(max_features=1000)
    result = evaluate(pipeline, train_df, test_df)

    output_path = tmp_path / "reports" / "confusion_matrix.png"
    save_confusion_matrix_plot(result, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


# Confirm training-dir names map onto the documented rewrite_method identifiers.
def test_infer_rewrite_method_from_directory_name(tmp_path: Path) -> None:
    """Assert processed_chat_llm / processed_chat / processed map correctly."""

    # LLM intent-preserving DMs.
    assert infer_rewrite_method(tmp_path / "processed_chat_llm") == "llm_intent_v1"
    # Deterministic rule-based DMs.
    assert infer_rewrite_method(Path("data/processed_chat")) == "rule_based_v1"
    # Original email/SMS with no rewrite.
    assert infer_rewrite_method(Path("data/processed")) == "none"
    # A typo must not be silently treated as rule_based_v1.
    assert infer_rewrite_method(Path("data/processed_chat_old")) == "unknown"


# Confirm an empty candidate list fails instead of inventing a threshold.
def test_pick_operating_point_rejects_an_empty_grid() -> None:
    """Assert pick_operating_point raises when no VAL grid points exist."""

    with pytest.raises(ValueError, match="empty candidate list"):
        pick_operating_point([])


# Confirm the shared VAL rule prefers max scam recall among floor-feasible points.
def test_pick_operating_point_maximizes_scam_recall_when_floor_is_met() -> None:
    """Assert a high-scam-recall feasible point beats a higher-F1 infeasible one."""

    candidates = [
        {
            "threshold": 0.30,
            "legit_recall": 0.86,
            "scam_recall": 0.99,
            "scam_f1": 0.90,
        },
        {
            "threshold": 0.50,
            "legit_recall": 0.92,
            "scam_recall": 0.95,
            "scam_f1": 0.93,
        },
        {
            "threshold": 0.30,
            "legit_recall": 0.70,
            "scam_recall": 1.00,
            "scam_f1": 0.80,
        },
    ]
    chosen, reason, feasible = pick_operating_point(candidates, legit_recall_floor=0.85)
    assert feasible is True
    assert chosen["scam_recall"] == 0.99
    assert chosen["threshold"] == 0.30
    assert reason == "max_scam_recall_subject_to_legit_recall_floor"


# Confirm the shared VAL rule falls back to max scam F1 when the floor is impossible.
def test_pick_operating_point_falls_back_to_scam_f1_when_floor_infeasible() -> None:
    """Assert every point below the ham-recall floor yields the F1 fallback reason."""

    candidates = [
        {
            "threshold": 0.30,
            "legit_recall": 0.40,
            "scam_recall": 1.00,
            "scam_f1": 0.70,
        },
        {
            "threshold": 0.50,
            "legit_recall": 0.60,
            "scam_recall": 0.90,
            "scam_f1": 0.80,
        },
    ]
    chosen, reason, feasible = pick_operating_point(candidates, legit_recall_floor=0.85)
    assert feasible is False
    assert chosen["scam_f1"] == 0.80
    assert reason == "legit_recall_floor_infeasible_max_scam_f1"


# Confirm score_threshold_grid emits one candidate per documented threshold.
def test_score_threshold_grid_matches_the_documented_grid_length() -> None:
    """Assert the shared scorer walks DEFAULT_THRESHOLD_GRID exactly once each."""

    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.6, 0.9]
    candidates = score_threshold_grid(y_true, y_proba, DEFAULT_THRESHOLD_GRID, C=0.25)
    assert len(candidates) == len(DEFAULT_THRESHOLD_GRID)
    assert {row["threshold"] for row in candidates} == set(DEFAULT_THRESHOLD_GRID)
    assert all(row["C"] == 0.25 for row in candidates)


# Confirm the sweep VAL grid adds 0.20 and 0.25 below the published default.
def test_expanded_threshold_grid_includes_0_20_and_0_25() -> None:
    """Assert 0.20 and 0.25 are searched in addition to the documented 0.30..0.70 cuts."""

    # The expanded grid must be a strict superset of the published TF-IDF grid.
    assert set(DEFAULT_THRESHOLD_GRID).issubset(set(EXPANDED_THRESHOLD_GRID))
    # 20/100 and 25/100 are the two extra operating points requested for this sweep.
    assert 0.20 in EXPANDED_THRESHOLD_GRID
    # Keep 0.25 as an explicit member, not only 0.20.
    assert 0.25 in EXPANDED_THRESHOLD_GRID
    # Step size stays 0.05 so VAL search remains comparable to the published reports.
    steps = [
        round(EXPANDED_THRESHOLD_GRID[index + 1] - EXPANDED_THRESHOLD_GRID[index], 2)
        for index in range(len(EXPANDED_THRESHOLD_GRID) - 1)
    ]
    # Every adjacent pair must be 0.05 apart.
    assert set(steps) == {0.05}


# Confirm the sweep C grid widens both below 0.25 and above 4.0.
def test_widened_c_grid_is_a_strict_superset_of_the_published_grid() -> None:
    """Assert every published C is kept and extra weaker/stronger values are added."""

    # Published {0.25, 1.0, 4.0} must still be searched.
    assert set(DEFAULT_C_GRID).issubset(set(WIDENED_C_GRID))
    # Stronger regularization than the published minimum C=0.25.
    assert any(value < 0.25 for value in WIDENED_C_GRID)
    # Weaker regularization than the published maximum C=4.0.
    assert any(value > 4.0 for value in WIDENED_C_GRID)
    # The widened grid must be strictly larger, not a rename of the published set.
    assert len(WIDENED_C_GRID) > len(DEFAULT_C_GRID)


# Confirm hyperparameters JSON round-trips through the report helpers.
def test_hyperparameters_round_trip_through_json_mapping() -> None:
    """Assert chat-eval can rebuild the same knobs train_baseline.py wrote."""

    original = BaselineHyperparameters(
        max_features=25_000,
        ngram_min=1,
        ngram_max=3,
        min_df=3,
        max_df=0.95,
        sublinear_tf=False,
        use_idf=False,
        stop_words="english",
        strip_accents=None,
        url_features=False,
        class_weight=None,
        solver="liblinear",
        max_iter=500,
        random_state=7,
    )
    payload = {"hyperparameters": hyperparameters_to_dict(original)}
    restored = hyperparameters_from_mapping(payload)
    assert restored == original


# Confirm a published-style report without a hyperparameters block still loads.
def test_hyperparameters_from_mapping_defaults_published_recipe() -> None:
    """Assert old baseline_metrics.json (max_features + url_features only) still works."""

    restored = hyperparameters_from_mapping({"max_features": 50000, "url_features": True})
    assert restored.max_features == 50_000
    assert restored.ngram_max == 2
    assert restored.url_features is True
    assert restored.class_weight == "balanced"
    assert restored.solver == "lbfgs"


# Confirm OFAT can drop the URL branch and still fit.
def test_build_pipeline_without_url_features_still_fits(
    synthetic_processed_dir: Path,
) -> None:
    """Assert url_features=False yields a TF-IDF-only FeatureUnion that trains."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, _val_df, test_df = stratified_split(combined, random_state=42)
    knobs = BaselineHyperparameters(max_features=1000, url_features=False)
    pipeline = build_pipeline(max_features=1000, hyperparameters=knobs)
    result = evaluate(pipeline, train_df, test_df, threshold=0.5)
    transformer_names = [
        name for name, _transformer in pipeline.named_steps["features"].transformer_list
    ]
    assert transformer_names == ["tfidf"]
    assert result.test_rows == len(test_df)


# Confirm n-gram OFAT actually changes the vectorizer range.
def test_build_pipeline_honors_unigram_only_ngram_range(
    synthetic_processed_dir: Path,
) -> None:
    """Assert ngram_max=1 reaches TfidfVectorizer rather than staying at (1, 2)."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, _val_df, _test_df = stratified_split(combined, random_state=42)
    knobs = BaselineHyperparameters(max_features=1000, ngram_min=1, ngram_max=1)
    pipeline = build_pipeline(max_features=1000, hyperparameters=knobs)
    pipeline.fit(train_df["text"], train_df["label"])
    vectorizer = pipeline.named_steps["features"].transformer_list[0][1]
    assert vectorizer.ngram_range == (1, 1)


# Confirm tune_on_validation searches a caller-supplied widened C grid.
def test_tune_on_validation_records_a_custom_c_grid(
    synthetic_processed_dir: Path,
) -> None:
    """Assert the frozen C comes from the supplied grid, including values below 0.25."""

    combined = load_processed_corpora(synthetic_processed_dir)
    train_df, val_df, _test_df = stratified_split(combined, random_state=42)
    custom_c = (0.05, 0.25, 8.0)
    tuning = tune_on_validation(
        train_df,
        val_df,
        max_features=1000,
        C_grid=custom_c,
        threshold_grid=EXPANDED_THRESHOLD_GRID,
        legit_recall_floor=0.85,
        random_state=42,
    )
    assert tuning.C in custom_c
    assert tuning.grid_C == list(custom_c)
    assert 0.20 in tuning.grid_thresholds
    assert 0.25 in tuning.grid_thresholds

