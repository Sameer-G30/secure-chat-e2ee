"""Exercise word-BiLSTM helpers against tiny synthetic strings.

These tests never load the 71k corpus, never download GloVe/Hub weights,
and must pass on CPU. Vocab is built from the strings in this file.
Live training is scripts/train_lstm.py.
"""

# Import inspect so we can prove threshold search has no chat_eval path.
import inspect

# Import Path for temporary checkpoint directories.
from pathlib import Path

# Import numpy to build synthetic probability vectors for threshold tests.
import numpy as np

# Import pandas to pass schema-shaped frames into evaluate_from_proba.
import pandas as pd

# Import pytest for fixtures, monkeypatch, and raises helpers.
import pytest

# Import torch to compare parameter tensors and pin tests to CPU.
import torch

from secure_chat_ml.baseline import DEFAULT_THRESHOLD_GRID
from secure_chat_ml.lstm import (
    PAD_INDEX,
    UNK_INDEX,
    LstmHyperparameters,
    WordBiLstmClassifier,
    as_label_list,
    as_text_list,
    assert_not_chat_eval_path,
    balanced_class_weights,
    build_model,
    build_vocab,
    count_truncated_texts,
    encode_texts,
    evaluate_from_proba,
    fit_url_scaler,
    is_link_heavy,
    load_saved_classifier,
    predict_scam_proba,
    recommend_char_lstm_exploration,
    save_classifier,
    tokenize_text,
    train_model,
    transform_url_features,
    tune_threshold_on_validation,
)
from secure_chat_ml.url_features import URL_FEATURE_NAMES

# Repeated ham/scam strings so a tiny LSTM has something to fit.
_LEGITIMATE_TEXTS = [
    "let's grab lunch tomorrow at noon",
    "can you send me the meeting notes",
    "happy birthday, hope you have a great day",
    "thanks for helping me move last weekend",
] * 8
# Scam strings include one URL so URL-concat has a non-zero TRAIN row.
_SCAM_TEXTS = [
    "urgent: your account will be suspended, verify your password now",
    "you have won a prize, click https://192.0.2.1/login to claim",
    "your bank account has been locked, confirm your login immediately",
    "click here immediately to unlock your frozen account",
] * 8


# Pin LSTM tests to CPU so pytest never needs a GPU.
@pytest.fixture
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> torch.device:
    """Force resolve_training_device() to CPU/fp32 for the tiny test model."""

    # Build the same tuple the production helper returns on a CPU-only box.
    cpu = torch.device("cpu")

    # Replace the production CUDA probe with a CPU stub.
    def _cpu_only() -> tuple[torch.device, bool, str]:
        # Report that fp16 is unused, matching the real LSTM CPU path.
        return cpu, False, "cpu_fp32_fp16_requires_cuda"

    # Patch the helper used by train_model and predict_scam_proba.
    monkeypatch.setattr("secure_chat_ml.lstm.resolve_training_device", _cpu_only)
    # Return the device so tests can pass it explicitly too.
    return cpu


# Documented hyperparameters scaled down for a 1-epoch CPU toy run.
def _tiny_hyperparams() -> LstmHyperparameters:
    """Return hyperparameters that finish in under a second on CPU."""

    # Keep max_tokens and vocab tiny so encode/train stay millisecond-scale.
    return LstmHyperparameters(
        embed_dim=16,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        max_tokens=16,
        max_vocab_size=64,
        batch_size=8,
        eval_batch_size=8,
        learning_rate=1e-2,
        num_train_epochs=1,
        seed=42,
    )


# Confirm as_text_list / as_label_list accept Series, lists, and arrays.
def test_as_text_and_label_list_accept_series_and_lists() -> None:
    """Assert helper conversions preserve order and string/int types."""

    # pandas Series is the usual caller from load_processed_corpora.
    texts = pd.Series(["hello", "urgent"])
    # Matching binary labels from the schema.
    labels = pd.Series([0, 1])
    # Series of texts becomes a list of strings.
    assert as_text_list(texts) == ["hello", "urgent"]
    # Series of labels becomes a list of ints.
    assert as_label_list(labels) == [0, 1]
    # Plain Python lists also round-trip.
    assert as_text_list(["hello"]) == ["hello"]
    # numpy arrays of labels become Python ints.
    assert as_label_list(np.array([1])) == [1]


# Confirm the documented tokenizer splits punctuation from words.
def test_tokenize_text_splits_whitespace_and_punctuation() -> None:
    """Assert URLs explode into short tokens rather than one giant token."""

    # A URL should not remain a single token (that would hide it from UNK).
    tokens = tokenize_text("see https://example.com/login now")
    # The scheme letters are one alphanumeric run.
    assert "https" in tokens
    # Each slash is its own punctuation token.
    assert "/" in tokens
    # The login path segment survives as its own token.
    assert "login" in tokens
    # Empty text yields no tokens (encode_texts inserts PAD).
    assert tokenize_text("") == []


# Confirm vocab is built from TRAIN strings only; held-out tokens become UNK.
def test_vocab_is_built_from_train_only() -> None:
    """Assert a unique held-out token is absent from the TRAIN vocab."""

    # TRAIN contains only ordinary lunch words.
    vocab = build_vocab(["hello lunch"], max_vocab_size=32)
    # PAD and UNK occupy the reserved indices.
    assert vocab["<pad>"] == PAD_INDEX
    # UNK is always id 1.
    assert vocab["<unk>"] == UNK_INDEX
    # The TRAIN word is present.
    assert "lunch" in vocab
    # A token that never appeared in TRAIN must not enter the vocab.
    assert "supercalifragilistic" not in vocab
    # Encoding the held-out string must therefore emit UNK, not a new id.
    token_ids, lengths = encode_texts(["supercalifragilistic"], vocab, max_tokens=8)
    # Non-pad ids in the row should all be UNK.
    used = [int(i) for i in token_ids[0] if int(i) != PAD_INDEX]
    # Every real token mapped to UNK.
    assert used == [UNK_INDEX]
    # Length is 1 for the single OOV token.
    assert int(lengths[0]) == 1


# Confirm count_truncated_texts sees overflow when max_tokens is tiny.
def test_count_truncated_texts_detects_overflow() -> None:
    """Assert a long string counts as truncated when max_tokens is 2."""

    # Many tokens so the overflow counter can fire.
    long_text = "urgent account verify password click prize bank locked gift"
    # Overflow is detected before truncation.
    assert count_truncated_texts([long_text], max_tokens=2) == 1
    # A short string does not count as truncated.
    assert count_truncated_texts(["hi"], max_tokens=16) == 0


# Confirm encode_texts pads on the right and never emits length 0.
def test_encode_texts_pads_and_rejects_zero_length() -> None:
    """Assert empty messages become a single PAD token with length 1."""

    # Vocab from a one-word TRAIN set.
    vocab = build_vocab(["hello"], max_vocab_size=16)
    # Encode an empty DM that would otherwise break pack_padded_sequence.
    token_ids, lengths = encode_texts([""], vocab, max_tokens=8)
    # One row, padded to max_tokens.
    assert token_ids.shape == (1, 8)
    # True length is 1, not 0.
    assert int(lengths[0]) == 1
    # The only real slot is PAD.
    assert int(token_ids[0, 0]) == PAD_INDEX


# Confirm the classifier input width concatenates LSTM pooled states and URL features.
def test_url_concat_shape_matches_feature_names() -> None:
    """Assert Linear.in_features == 2 * hidden_size + len(URL_FEATURE_NAMES)."""

    # Tiny network used only for shape assertions.
    model = WordBiLstmClassifier(vocab_size=10, embed_dim=8, hidden_size=4, dropout=0.0)
    # Bidirectional last-state concat is 8, plus the frozen URL vector width.
    assert model.combined_dim == 8 + len(URL_FEATURE_NAMES)
    # The linear head must consume that concatenated width.
    assert model.classifier.in_features == model.combined_dim
    # Two logits for legitimate vs scam.
    assert model.classifier.out_features == 2


# Confirm URL features actually change logits for identical token ids.
def test_url_features_change_logits_when_tokens_match(cpu_device: torch.device) -> None:
    """Assert a non-zero URL vector is not ignored by the classification head."""

    # Tiny network with dropout off so the test is deterministic.
    model = WordBiLstmClassifier(vocab_size=10, embed_dim=8, hidden_size=4, dropout=0.0)
    # Force a non-zero has_url weight so random init cannot hide the concat.
    with torch.no_grad():
        # has_url is the first concatenated URL column (after 2 * hidden_size).
        model.classifier.weight[:, 2 * model.hidden_size] = 1.0
    # Eval mode so dropout (already 0) stays off.
    model.eval()
    # One padded row of token id 2 (not PAD/UNK) of length 4.
    ids = torch.full((1, 4), 2, dtype=torch.long)
    # Packing length matches the four real tokens.
    lengths = torch.tensor([4], dtype=torch.long)
    # All-zero URL vector (has_url=0 message).
    zeros = torch.zeros(1, len(URL_FEATURE_NAMES), dtype=torch.float32)
    # Flip has_url so the concat block differs.
    ones = zeros.clone()
    # First URL_FEATURE_NAMES entry is has_url.
    ones[0, 0] = 1.0
    # Score both URL vectors without updating weights.
    with torch.no_grad():
        # Logits with the zero URL vector.
        logits_zero = model(ids, lengths, zeros)
        # Logits with has_url=1 and the same tokens.
        logits_url = model(ids, lengths, ones)
    # If concat is wired, the URL column must move the head (random init ≠ 0).
    assert logits_zero.shape == (1, 2)
    # The two forward passes must not be identical.
    assert not torch.allclose(logits_zero, logits_url)


# Confirm link-free messages still produce a URL vector of the right width.
def test_no_url_messages_keep_zero_vector_and_are_not_dropped() -> None:
    """Assert lunch DMs keep has_url=0 rows rather than being filtered out."""

    # Fit the scaler on a mix of URL and no-URL TRAIN rows.
    scaler = fit_url_scaler(
        ["lunch tomorrow", "verify https://192.0.2.1/login"],
    )
    # Transform a link-free message; it must remain a row, not dropped.
    scaled = transform_url_features(["lunch tomorrow"], scaler)
    # One row, one column per URL_FEATURE_NAMES entry.
    assert scaled.shape == (1, len(URL_FEATURE_NAMES))
    # A two-row batch including a URL message must also keep both rows.
    both = transform_url_features(
        ["lunch tomorrow", "see https://example.com/doc"], scaler
    )
    # Two messages in, two rows out.
    assert both.shape == (2, len(URL_FEATURE_NAMES))


# Confirm is_link_heavy flags URL-dominated DMs but not a short docs link.
def test_is_link_heavy_requires_url_dominated_text() -> None:
    """Assert a long URL-only DM is link-heavy and a lunch+docs DM is not."""

    # Almost the entire message is the URL characters.
    url_only = "https://192.0.2.1/login"
    # A legitimate DM that merely attaches a docs link inside longer prose.
    lunch_docs = (
        "hey, let's grab lunch tomorrow at noon near the office and catch up "
        "about the weekend hike plus the birthday plans for sam, and if you "
        "have a second please skim the shared notes here "
        "https://docs.google.com/x"
    )
    # URL-only should trip the 30% character-fraction rule.
    assert is_link_heavy(url_only) is True
    # Ordinary prose plus one https link should not be link-heavy.
    assert is_link_heavy(lunch_docs) is False
    # No URL at all is never link-heavy.
    assert is_link_heavy("hi mom") is False


# Confirm VAL threshold selection returns a documented grid value.
def test_tune_threshold_on_validation_returns_a_grid_value() -> None:
    """Assert the frozen cut comes from DEFAULT_THRESHOLD_GRID, not chat_eval."""

    # Balanced synthetic labels.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Well-separated probabilities so the floor is feasible.
    y_proba = np.array([0.05, 0.10, 0.20, 0.40, 0.80, 0.90, 0.95, 0.99])
    # Search only the documented grid.
    result = tune_threshold_on_validation(
        y_true,
        y_proba,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        legit_recall_floor=0.85,
    )
    # The chosen cut must be one of the documented values.
    assert result.threshold in DEFAULT_THRESHOLD_GRID
    # VAL row count matches the synthetic vector.
    assert result.val_rows == 8
    # Reason must be one of the two documented selection outcomes.
    assert result.selection_reason in {
        "max_scam_recall_subject_to_legit_recall_floor",
        "legit_recall_floor_infeasible_max_scam_f1",
    }
    # The recorded grid is exactly the documented default.
    assert result.grid_thresholds == list(DEFAULT_THRESHOLD_GRID)


# Confirm threshold search cannot accept a chat_eval filesystem path.
def test_tune_threshold_signature_has_no_chat_eval_path() -> None:
    """Assert tune_threshold_on_validation only takes labels and probabilities."""

    # Inspect the public helper used for VAL search.
    signature = inspect.signature(tune_threshold_on_validation)
    # Parameter names must not include a corpus path the 200-row file could fill.
    names = set(signature.parameters)
    # Only labels, probabilities, and grid/floor knobs are allowed.
    assert "y_true" in names
    # Probabilities are precomputed; the helper never loads a CSV.
    assert "y_proba" in names
    # No path-like argument that could point at data/chat_eval/.
    assert "path" not in names
    # Explicit chat_eval name must not exist either.
    assert "chat_eval" not in names
    # texts/corpus would also be a leak surface; they must stay out.
    assert "texts" not in names


# Confirm a mis-pointed chat_eval directory is refused as a training source.
def test_assert_not_chat_eval_path_refuses_locked_eval_dir(tmp_path: Path) -> None:
    """Assert data/chat_eval cannot be used to fit vocab or the LSTM."""

    # A nested chat_eval directory must be rejected.
    with pytest.raises(ValueError, match="chat_style_eval_training_allowed"):
        # Point at a path whose parts include chat_eval.
        assert_not_chat_eval_path(tmp_path / "data" / "chat_eval")


# Confirm an impossible ham-recall floor records the documented F1 fallback.
def test_tune_threshold_on_validation_falls_back_when_floor_infeasible() -> None:
    """Assert legit_recall_floor=1.01 records the F1 fallback reason."""

    # Tiny label/probability vectors.
    y_true = np.array([0, 0, 1, 1])
    # Probabilities that cannot achieve legitimate recall 1.01.
    y_proba = np.array([0.2, 0.3, 0.8, 0.9])
    # Search with an infeasible floor so the fallback path is exercised.
    result = tune_threshold_on_validation(
        y_true,
        y_proba,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        legit_recall_floor=1.01,
    )
    # The floor was not met.
    assert result.floor_feasible is False
    # The documented F1 fallback reason is recorded.
    assert result.selection_reason == "legit_recall_floor_infeasible_max_scam_f1"


# Confirm evaluate_from_proba builds a 2x2 matrix whose cells sum to n.
def test_evaluate_from_proba_reports_a_well_formed_confusion_matrix() -> None:
    """Assert thresholded probabilities produce schema-shaped metrics."""

    # Two ham and two scam gold labels.
    y_true = pd.Series([0, 0, 1, 1])
    # Probabilities that separate cleanly at 0.5.
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    # Apply a frozen 0.5 cut.
    result = evaluate_from_proba(y_true, y_proba, threshold=0.5, train_rows=10)
    # train_rows is a stamp, not inferred from y_true.
    assert result.train_rows == 10
    # Four scored rows.
    assert result.test_rows == 4
    # Threshold is recorded.
    assert result.threshold == 0.5
    # Confusion-matrix cells sum to the scored row count.
    assert sum(sum(row) for row in result.confusion_matrix) == 4
    # Perfect separation at 0.5.
    assert result.confusion_matrix == [[2, 0], [0, 2]]


# Confirm balanced class weights are a length-2 CPU tensor.
def test_balanced_class_weights_are_length_two() -> None:
    """Assert TRAIN-only class weights have one entry per schema label."""

    # Three TRAIN labels with imbalance.
    weights = balanced_class_weights([0, 0, 1])
    # One weight per class.
    assert weights.shape == (2,)
    # float32 CPU tensor, matching DistilBERT.
    assert weights.dtype == torch.float32
    # Weights stay on CPU until the training loop moves them.
    assert weights.device.type == "cpu"


# Confirm a 1-epoch tiny train runs and scoring does not update weights.
def test_train_then_predict_does_not_update_weights(cpu_device: torch.device) -> None:
    """Assert predict_scam_proba leaves parameters unchanged after training."""

    # Build TRAIN vocab from the synthetic strings in this file.
    texts = _LEGITIMATE_TEXTS + _SCAM_TEXTS
    # Matching labels: 0 for ham repeats, 1 for scam repeats.
    labels = [0] * len(_LEGITIMATE_TEXTS) + [1] * len(_SCAM_TEXTS)
    # Tiny knobs so the loop finishes in milliseconds on CPU.
    hyperparams = _tiny_hyperparams()
    # Vocab from these strings only (no 71k corpus).
    token_to_id = build_vocab(texts, max_vocab_size=hyperparams.max_vocab_size)
    # URL scaler from these strings only.
    url_scaler = fit_url_scaler(texts)
    # Untrained tiny network.
    model = build_model(vocab_size=len(token_to_id), hyperparams=hyperparams)
    # One-epoch TRAIN-only fit.
    trained = train_model(
        model,
        texts,
        labels,
        token_to_id,
        url_scaler,
        hyperparams=hyperparams,
    )
    # Snapshot weights after training, before predict.
    before = [param.detach().clone() for param in trained.parameters()]
    # Score a handful of rows; this must not call backward().
    proba = predict_scam_proba(
        trained,
        texts[:8],
        token_to_id,
        url_scaler,
        max_tokens=hyperparams.max_tokens,
        batch_size=4,
        device=cpu_device,
    )
    # One probability per scored text.
    assert proba.shape == (8,)
    # Softmax outputs stay in [0, 1].
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    # Parameters after predict must equal the snapshot.
    after = list(trained.parameters())
    # Pair each snapshot tensor with the live parameter.
    for left, right in zip(before, after, strict=True):
        # Bit-exact equality: scoring is no_grad eval.
        assert torch.equal(left, right)


# Confirm save/load round-trips a tiny checkpoint from disk, not the Hub.
def test_save_and_load_classifier_round_trip(
    cpu_device: torch.device, tmp_path: Path
) -> None:
    """Assert a saved tiny model reloads and produces the same probabilities."""

    # Two short strings, one with a URL so scaler stats are non-degenerate.
    texts = ["let's grab lunch tomorrow", "urgent verify https://192.0.2.1/login"]
    # Tiny knobs matching the other CPU tests.
    hyperparams = _tiny_hyperparams()
    # Vocab from these two strings.
    token_to_id = build_vocab(texts, max_vocab_size=hyperparams.max_vocab_size)
    # URL scaler from these two strings.
    url_scaler = fit_url_scaler(texts)
    # Untrained tiny network (weights still random, but round-trip must match).
    model = build_model(vocab_size=len(token_to_id), hyperparams=hyperparams)
    # Eval mode for deterministic dropout-off scoring.
    model.eval()
    # Score before saving.
    original = predict_scam_proba(
        model,
        texts,
        token_to_id,
        url_scaler,
        max_tokens=hyperparams.max_tokens,
        batch_size=2,
        device=cpu_device,
    )
    # Write model.pt + meta.json under a temp directory.
    checkpoint = tmp_path / "checkpoint"
    # Persist weights, vocab, scaler, and a dummy VAL threshold.
    save_classifier(
        model,
        token_to_id,
        url_scaler,
        checkpoint,
        hyperparams=hyperparams,
        threshold=0.30,
    )
    # Reload without touching the Hub or the 71k corpus.
    loaded_model, loaded_vocab, loaded_scaler, loaded_hp, loaded_thr = (
        load_saved_classifier(checkpoint)
    )
    # Sidecar threshold must round-trip.
    assert loaded_thr == pytest.approx(0.30)
    # Hyperparameter max_tokens must round-trip.
    assert loaded_hp.max_tokens == hyperparams.max_tokens
    # Reloaded vocab must match the TRAIN map.
    assert loaded_vocab == token_to_id
    # Score after load; probabilities must match the pre-save vector.
    reloaded = predict_scam_proba(
        loaded_model,
        texts,
        loaded_vocab,
        loaded_scaler,
        max_tokens=loaded_hp.max_tokens,
        batch_size=2,
        device=cpu_device,
    )
    # Allow tiny float tolerance for CPU softmax.
    np.testing.assert_allclose(original, reloaded, rtol=1e-5, atol=1e-5)


# Confirm load_saved_classifier fails loudly when training has not been run.
def test_load_saved_classifier_requires_a_checkpoint(tmp_path: Path) -> None:
    """Assert a missing model dir tells the operator to run train_lstm.py."""

    # Path that does not exist yet.
    missing = tmp_path / "models" / "lstm"
    # Missing checkpoint must mention the training script, not pytest.
    with pytest.raises(FileNotFoundError, match="train_lstm.py"):
        # Attempt to load from the empty directory.
        load_saved_classifier(missing)


# Confirm predict_scam_proba returns an empty array for an empty input list.
def test_predict_scam_proba_on_empty_input(cpu_device: torch.device) -> None:
    """Assert scoring zero rows does not crash and returns a 1-d empty vector."""

    # Tiny vocab so we can construct a network.
    vocab = build_vocab(["hello"], max_vocab_size=16)
    # Scaler fit on the same one string.
    scaler = fit_url_scaler(["hello"])
    # Tiny network.
    model = build_model(vocab_size=len(vocab), hyperparams=_tiny_hyperparams())
    # Score an empty list.
    proba = predict_scam_proba(
        model,
        [],
        vocab,
        scaler,
        max_tokens=16,
        batch_size=4,
        device=cpu_device,
    )
    # Shape (0,) rather than a scalar or 2-d empty array.
    assert proba.shape == (0,)


# Confirm the char-LSTM rule says do-not-explore when the model is a solid third baseline.
def test_recommend_char_lstm_do_not_explore_for_reasonable_third_baseline() -> None:
    """Assert A/B/C fail when TEST is near DistilBERT and chat FPs beat TF-IDF."""

    # Numbers shaped like a usable third baseline, not a link-heavy collapse.
    result = recommend_char_lstm_exploration(
        chat_eval_fn_lstm=8,
        chat_eval_fn_tfidf=0,
        test_fn_lstm=70,
        test_fn_distilbert=60,
        test_scam_recall_lstm=0.980,
        test_scam_recall_distilbert=0.983,
        test_scam_recall_tfidf=0.992,
        chat_eval_scam_recall_lstm=0.92,
        chat_eval_ham_warned_lstm=12,
        chat_eval_ham_warned_tfidf=70,
        extra_fn_chat_eval_url_related=2,
        extra_fn_test_url_related=10,
        lstm_url_scam_recall_chat=0.95,
        lstm_url_scam_recall_test=0.99,
        tfidf_url_scam_recall_chat=None,
        tfidf_url_scam_recall_test=None,
        tfidf_scam_recall_chat=1.0,
        chat_eval_fn_url=2,
        chat_eval_fn_no_url=6,
        test_fn_url=10,
        test_fn_no_url=60,
    )
    # Chat extras are 8 (< 10) and TEST recall gap is 0.003 (< 0.05) so A is false.
    assert result["criterion_a"] is False
    # Misses are mostly no-URL social-engineering.
    assert result["mostly_no_url"] is True
    # Verdict must refuse char LSTM.
    assert result["verdict"] == "do_not_explore_char_lstm"


# Confirm the char-LSTM rule says explore only when A, B, and C all hold.
def test_recommend_char_lstm_explore_when_a_b_c_all_hold() -> None:
    """Assert implement-now fires for link-heavy extra misses and a URL-recall gap."""

    # Chat-eval misses 20 extra scams vs TF-IDF, mostly URL-bearing.
    result = recommend_char_lstm_exploration(
        chat_eval_fn_lstm=20,
        chat_eval_fn_tfidf=0,
        test_fn_lstm=400,
        test_fn_distilbert=60,
        test_scam_recall_lstm=0.88,
        test_scam_recall_distilbert=0.983,
        test_scam_recall_tfidf=0.992,
        chat_eval_scam_recall_lstm=0.80,
        chat_eval_ham_warned_lstm=80,
        chat_eval_ham_warned_tfidf=70,
        extra_fn_chat_eval_url_related=16,
        extra_fn_test_url_related=300,
        lstm_url_scam_recall_chat=0.70,
        lstm_url_scam_recall_test=0.80,
        tfidf_url_scam_recall_chat=None,
        tfidf_url_scam_recall_test=None,
        tfidf_scam_recall_chat=1.0,
        chat_eval_fn_url=16,
        chat_eval_fn_no_url=4,
        test_fn_url=300,
        test_fn_no_url=100,
    )
    # A: 20 extra chat FNs and a large TEST recall gap.
    assert result["criterion_a"] is True
    # B: 16/20 extra FNs are URL-related.
    assert result["criterion_b"] is True
    # C: URL-bearing recall well below TF-IDF overall.
    assert result["criterion_c"] is True
    # Conjunction plus no vetoes → explore (still not implemented in this pass).
    assert result["verdict"] == "explore_char_lstm"
    # implement_now matches the conjunction.
    assert result["implement_now"] is True
