"""Exercise DistilBERT helpers against a tiny random model, never the Hub.

These tests construct DistilBERT from a local vocab file and a 1-layer
config. They must pass with HF_HUB_OFFLINE=1 and must not download
`distilbert-base-uncased`. Live training is scripts/train_distilbert.py.
"""

# Import Path for temporary vocab and checkpoint directories.
from pathlib import Path

# Import numpy to build synthetic probability vectors for threshold tests.
import numpy as np

# Import pandas to pass schema-shaped frames into evaluate_from_proba.
import pandas as pd

# Import pytest for fixtures, monkeypatch, and raises helpers.
import pytest

# Import torch to compare parameter tensors and pin tests to CPU.
import torch

# Import HuggingFace pieces used only to build the local tiny stand-in.
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

from secure_chat_ml.baseline import DEFAULT_THRESHOLD_GRID
from secure_chat_ml.distilbert import (
    DISTILBERT_EXPANDED_THRESHOLD_GRID,
    DistilBertHyperparameters,
    as_label_list,
    as_text_list,
    balanced_class_weights,
    count_truncated_texts,
    evaluate_from_proba,
    fine_tune,
    load_saved_classifier,
    predict_scam_proba,
    save_classifier,
    tokenize_texts,
    tune_threshold_on_validation,
)

# Repeated ham/scam strings so a 1-layer toy model has something to fit.
_LEGITIMATE_TEXTS = [
    "let's grab lunch tomorrow at noon",
    "can you send me the meeting notes",
    "happy birthday, hope you have a great day",
    "thanks for helping me move last weekend",
] * 8
_SCAM_TEXTS = [
    "urgent: your account will be suspended, verify your password now",
    "you have won a prize, click this link to claim your reward",
    "your bank account has been locked, confirm your login immediately",
    "click here immediately to unlock your frozen account",
] * 8


# Pin DistilBERT tests to CPU so pytest never needs a GPU or Hub download.
@pytest.fixture
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> torch.device:
    """Force resolve_training_device() to CPU/fp32 for the tiny test model."""

    # Build the same tuple the production helper returns on a CPU-only box.
    cpu = torch.device("cpu")

    # Replace the production CUDA probe with a CPU stub.
    def _cpu_only() -> tuple[torch.device, bool, str]:
        # Report that fp16 is unavailable, matching the real CPU path.
        return cpu, False, "cpu_fp32_fp16_requires_cuda"

    # Patch the helper used by fine_tune and predict_scam_proba.
    monkeypatch.setattr("secure_chat_ml.distilbert.resolve_training_device", _cpu_only)
    # Return the device so tests can pass it explicitly too.
    return cpu


# Write a tiny WordPiece vocab and build a 1-layer random DistilBERT.
@pytest.fixture
def tiny_classifier(
    tmp_path: Path,
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
    """Return (model, tokenizer) built entirely from local files."""

    # Include DistilBERT special tokens plus a handful of ham/scam cue words.
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        "lunch",
        "meeting",
        "thanks",
        "birthday",
        "urgent",
        "account",
        "verify",
        "password",
        "click",
        "prize",
        "bank",
        "locked",
        "gift",
        "the",
        "a",
        "to",
        "your",
        "now",
        "will",
        "be",
        "let",
        "grab",
        "tomorrow",
        "send",
        "notes",
        "help",
        "move",
        "won",
        "link",
        "claim",
        "reward",
        "confirm",
        "login",
        "unlock",
        "frozen",
    ]
    # Write vocab.txt in the BERT one-token-per-line format.
    vocab_path = tmp_path / "vocab.txt"
    # Join with newlines so DistilBertTokenizer can parse the file.
    vocab_path.write_text("\n".join(vocab_tokens) + "\n")
    # Load the tokenizer from the local vocab; this must not hit the Hub.
    tokenizer = DistilBertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    # Build a 1-layer, 32-dim DistilBERT that fits in CPU RAM in milliseconds.
    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        hidden_dim=64,
        n_layers=1,
        n_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
        id2label={0: "legitimate", 1: "scam"},
        label2id={"legitimate": 0, "scam": 1},
    )
    # Initialize weights randomly; tests only require the training loop to run.
    model = DistilBertForSequenceClassification(config)
    # Return the pair the production helpers expect.
    return model, tokenizer


# Documented hyperparameters scaled down for a 1-epoch CPU toy run.
def _tiny_hyperparams() -> DistilBertHyperparameters:
    """Return hyperparameters that finish in under a second on CPU."""

    # Keep max_length under the tiny config's max_position_embeddings.
    return DistilBertHyperparameters(
        model_name="tiny-local",
        max_length=32,
        train_batch_size=8,
        eval_batch_size=8,
        learning_rate=5e-4,
        num_train_epochs=1,
        warmup_ratio=0.0,
        seed=42,
    )


# Confirm as_text_list / as_label_list accept Series, lists, and arrays.
def test_as_text_and_label_list_accept_series_and_lists() -> None:
    """Assert helper conversions preserve order and string/int types."""

    texts = pd.Series(["hello", "urgent"])
    labels = pd.Series([0, 1])
    assert as_text_list(texts) == ["hello", "urgent"]
    assert as_label_list(labels) == [0, 1]
    assert as_text_list(["hello"]) == ["hello"]
    assert as_label_list(np.array([1])) == [1]


# Confirm balanced class weights are a length-2 CPU tensor.
def test_balanced_class_weights_are_length_two() -> None:
    """Assert TRAIN-only class weights have one entry per schema label."""

    weights = balanced_class_weights([0, 0, 1])
    assert weights.shape == (2,)
    assert weights.dtype == torch.float32
    assert weights.device.type == "cpu"


# Confirm tokenize_texts truncates without padding to max_length.
def test_tokenize_texts_does_not_pad_to_max_length(tiny_classifier) -> None:
    """Assert unpadded encodings are shorter than max_length for short DMs."""

    _model, tokenizer = tiny_classifier
    encoded = tokenize_texts(tokenizer, ["hi"], max_length=32)
    assert len(encoded["input_ids"]) == 1
    assert len(encoded["input_ids"][0]) < 32
    assert len(encoded["input_ids"][0]) == len(encoded["attention_mask"][0])


# Confirm count_truncated_texts sees overflow when max_length is tiny.
def test_count_truncated_texts_detects_overflow(tiny_classifier) -> None:
    """Assert a long string counts as truncated when max_length is 4."""

    _model, tokenizer = tiny_classifier
    long_text = "urgent account verify password click prize bank locked gift"
    assert count_truncated_texts(tokenizer, [long_text], max_length=4) == 1
    assert count_truncated_texts(tokenizer, ["hi"], max_length=32) == 0


# Confirm the DistilBERT sweep grid adds 0.20 and 0.25 below the TF-IDF default.
def test_expanded_threshold_grid_includes_0_20_and_0_25() -> None:
    """Assert 0.20 and 0.25 are searched in addition to the documented 0.30..0.70 cuts."""

    # The expanded grid must be a strict superset of the TF-IDF / Slice 5 default.
    assert set(DEFAULT_THRESHOLD_GRID).issubset(set(DISTILBERT_EXPANDED_THRESHOLD_GRID))
    # 20/100 and 25/100 are the two extra operating points requested for this sweep.
    assert 0.20 in DISTILBERT_EXPANDED_THRESHOLD_GRID
    # Keep 0.25 as an explicit member, not only 0.20.
    assert 0.25 in DISTILBERT_EXPANDED_THRESHOLD_GRID
    # Step size stays 0.05 so VAL search remains comparable to the published reports.
    expanded = DISTILBERT_EXPANDED_THRESHOLD_GRID
    steps = [
        round(expanded[index + 1] - expanded[index], 2)
        for index in range(len(expanded) - 1)
    ]
    # Every adjacent pair must be 0.05 apart.
    assert set(steps) == {0.05}


# Confirm VAL threshold selection returns a documented grid value.
def test_tune_threshold_on_validation_returns_a_grid_value() -> None:
    """Assert the frozen cut comes from DEFAULT_THRESHOLD_GRID, not chat_eval."""

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.10, 0.20, 0.40, 0.80, 0.90, 0.95, 0.99])
    result = tune_threshold_on_validation(
        y_true,
        y_proba,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        legit_recall_floor=0.85,
    )
    assert result.threshold in DEFAULT_THRESHOLD_GRID
    assert result.val_rows == 8
    assert result.selection_reason in {
        "max_scam_recall_subject_to_legit_recall_floor",
        "legit_recall_floor_infeasible_max_scam_f1",
    }
    assert result.grid_thresholds == list(DEFAULT_THRESHOLD_GRID)


# Confirm an impossible ham-recall floor records the documented F1 fallback.
def test_tune_threshold_on_validation_falls_back_when_floor_infeasible() -> None:
    """Assert legit_recall_floor=1.01 records the F1 fallback reason."""

    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.8, 0.9])
    result = tune_threshold_on_validation(
        y_true,
        y_proba,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        legit_recall_floor=1.01,
    )
    assert result.floor_feasible is False
    assert result.selection_reason == "legit_recall_floor_infeasible_max_scam_f1"


# Confirm evaluate_from_proba builds a 2x2 matrix whose cells sum to n.
def test_evaluate_from_proba_reports_a_well_formed_confusion_matrix() -> None:
    """Assert thresholded probabilities produce schema-shaped metrics."""

    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    result = evaluate_from_proba(y_true, y_proba, threshold=0.5, train_rows=10)
    assert result.train_rows == 10
    assert result.test_rows == 4
    assert result.threshold == 0.5
    assert sum(sum(row) for row in result.confusion_matrix) == 4
    assert result.confusion_matrix == [[2, 0], [0, 2]]


# Confirm a 1-epoch tiny fine-tune runs and scoring does not update weights.
def test_fine_tune_then_predict_does_not_update_weights(
    tiny_classifier, cpu_device, tmp_path: Path
) -> None:
    """Assert predict_scam_proba leaves parameters unchanged after training."""

    model, tokenizer = tiny_classifier
    texts = _LEGITIMATE_TEXTS + _SCAM_TEXTS
    labels = [0] * len(_LEGITIMATE_TEXTS) + [1] * len(_SCAM_TEXTS)
    trained = fine_tune(
        model,
        tokenizer,
        texts,
        labels,
        output_dir=tmp_path / "scratch",
        hyperparams=_tiny_hyperparams(),
    )
    before = [param.detach().clone() for param in trained.parameters()]
    proba = predict_scam_proba(
        trained,
        tokenizer,
        texts[:8],
        max_length=32,
        batch_size=4,
        device=cpu_device,
        use_fp16=False,
    )
    assert proba.shape == (8,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    after = list(trained.parameters())
    for left, right in zip(before, after, strict=True):
        assert torch.equal(left, right)


# Confirm save/load round-trips a tiny checkpoint from disk, not the Hub.
def test_save_and_load_classifier_round_trip(tiny_classifier, cpu_device, tmp_path: Path) -> None:
    """Assert a saved tiny model reloads and produces the same probabilities."""

    model, tokenizer = tiny_classifier
    texts = ["let's grab lunch tomorrow", "urgent verify your password now"]
    original = predict_scam_proba(
        model,
        tokenizer,
        texts,
        max_length=32,
        batch_size=2,
        device=cpu_device,
        use_fp16=False,
    )
    checkpoint = tmp_path / "checkpoint"
    save_classifier(model, tokenizer, checkpoint)
    loaded_model, loaded_tokenizer = load_saved_classifier(checkpoint)
    reloaded = predict_scam_proba(
        loaded_model,
        loaded_tokenizer,
        texts,
        max_length=32,
        batch_size=2,
        device=cpu_device,
        use_fp16=False,
    )
    np.testing.assert_allclose(original, reloaded, rtol=1e-5, atol=1e-5)


# Confirm load_saved_classifier fails loudly when training has not been run.
def test_load_saved_classifier_requires_a_checkpoint(tmp_path: Path) -> None:
    """Assert a missing model dir tells the operator to run train_distilbert.py."""

    missing = tmp_path / "models" / "distilbert"
    with pytest.raises(FileNotFoundError, match="train_distilbert.py"):
        load_saved_classifier(missing)


# Confirm predict_scam_proba returns an empty array for an empty input list.
def test_predict_scam_proba_on_empty_input(tiny_classifier, cpu_device) -> None:
    """Assert scoring zero rows does not crash and returns a 1-d empty vector."""

    model, tokenizer = tiny_classifier
    proba = predict_scam_proba(
        model,
        tokenizer,
        [],
        max_length=32,
        batch_size=4,
        device=cpu_device,
        use_fp16=False,
    )
    assert proba.shape == (0,)
