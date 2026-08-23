"""Exercise ONNX export helpers on tiny models, never the 71k corpus.

These tests do not load DistilBERT-base, do not read chat_eval for fitting,
and do not overwrite ml/reports/*.json. Live export is scripts/export_onnx_web.py.
"""

# Import json so catalog/manifest helpers can be round-tripped in tmp_path.
import json

# Import math for a local sigmoid used to cross-check the logistic ONNX head.
import math

# Import Path for temporary export directories.
from pathlib import Path

# Import numpy for dense feature rows fed to the logistic ONNX graph.
import numpy as np

# Import pandas to build a tiny schema-shaped TRAIN frame.
# Import pytest for tmp_path and skip-friendly assertions.
import pytest

# Import torch to compare PyTorch logits with ORT CPU logits.
import torch

# Import the tiny DistilBERT fixtures already used by test_distilbert.py.
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer

from secure_chat_ml.baseline import build_pipeline
from secure_chat_ml.lstm import (
    LstmHyperparameters,
    build_model,
    build_vocab,
    fit_url_scaler,
    tokenize_text,
    train_model,
    transform_url_features,
)
from secure_chat_ml.onnx_export import (
    DistilBertLogitsWrapper,
    ExportableWordBiLstm,
    build_logistic_head_onnx,
    encode_lstm_unpadded,
    extract_tfidf_export_payload,
    force_eager_attention,
    run_logistic_onnx,
    write_tfidf_sidecars,
    write_wordpiece_vocab,
)
from secure_chat_ml.onnx_web_catalog import CHECKPOINT_CATALOG, ONNX_WEB_FIXTURES
from secure_chat_ml.url_features import URL_FEATURE_NAMES

# Repeated ham/scam strings so TF-IDF min_df=2 still has a vocabulary.
_LEGITIMATE_TEXTS = [
    "let's grab lunch tomorrow at noon",
    "can you send me the meeting notes",
    "happy birthday, hope you have a great day",
    "thanks for helping me move last weekend",
] * 8
_SCAM_TEXTS = [
    "urgent: your account will be suspended, verify your password now",
    "you have won a prize, click https://192.0.2.1/login to claim",
    "your bank account has been locked, confirm your login immediately",
    "click here immediately to unlock your frozen account",
] * 8


# Confirm the six-way load order is DistilBERT → LSTM → TF-IDF as specified.
def test_catalog_load_order_is_fixed() -> None:
    """Assert load_order 1..6 matches the Slice 6 sequential check."""

    # Sort a copy so a catalog edit that breaks order fails this test.
    ordered = sorted(CHECKPOINT_CATALOG, key=lambda row: int(row["load_order"]))
    # The ids are the contract the TypeScript load-check duplicates.
    ids = [str(row["id"]) for row in ordered]
    # Fail if anyone reorders the six checkpoints.
    assert ids == [
        "distilbert_best",
        "distilbert_default",
        "lstm_best",
        "lstm_default",
        "tfidf_best",
        "tfidf_default",
    ]


# Confirm browser fixtures are short DMs and not the locked chat-eval path.
def test_fixtures_are_not_chat_eval_and_cover_url_and_no_url() -> None:
    """Assert four fixtures mix ham/scam with and without URLs."""

    # Exactly four DMs keep the tab check cheap.
    assert len(ONNX_WEB_FIXTURES) == 4
    # Gold labels must stay 0=legitimate, 1=scam.
    labels = {int(row["gold_label"]) for row in ONNX_WEB_FIXTURES}
    # Both classes must appear so banner on and off are exercised.
    assert labels == {0, 1}
    # Concatenate texts to check URL presence without importing chat_eval.
    joined = " ".join(str(row["text"]) for row in ONNX_WEB_FIXTURES)
    # At least one https fixture so URL features are non-zero.
    assert "https://" in joined
    # The locked eval filename must not be referenced as a training source.
    for row in ONNX_WEB_FIXTURES:
        # Fixture ids are short names, not chat_style_eval_v1 rows.
        assert "chat_style_eval" not in str(row["id"])


# Confirm export source never converts TfidfVectorizer with skl2onnx.
def test_export_module_does_not_skl2onnx_tfidf() -> None:
    """Assert onnx_export.py does not import skl2onnx (A5)."""

    # Locate the export module next to this test file's package.
    source_path = Path(__file__).resolve().parents[1] / "src" / "secure_chat_ml" / "onnx_export.py"
    # Read the module as text for a static ban.
    source = source_path.read_text(encoding="utf-8")
    # skl2onnx of TfidfVectorizer is the forbidden browser path.
    assert "import skl2onnx" not in source
    # from-import is equally forbidden; the docstring may mention the ban.
    assert "from skl2onnx" not in source
    # The allowed path is a hand-built Gemm+Sigmoid graph.
    assert "Gemm" in source


# Logistic ONNX P(scam) must match sklearn predict_proba on a tiny pipeline.
def test_logistic_onnx_matches_sklearn_predict_proba(tmp_path: Path) -> None:
    """Assert Gemm+Sigmoid matches LogisticRegression.predict_proba[:, 1]."""

    # Build a tiny labeled frame so FeatureUnion can fit.
    texts = _LEGITIMATE_TEXTS + _SCAM_TEXTS
    # Binary labels matching the schema.
    labels = [0] * len(_LEGITIMATE_TEXTS) + [1] * len(_SCAM_TEXTS)
    # Tiny vocab so the test stays millisecond-scale.
    pipeline = build_pipeline(max_features=64, C=1.0)
    # Fit TRAIN-only on the synthetic strings.
    pipeline.fit(texts, labels)
    # Pull coef/intercept/vocab from the fitted pipeline.
    payload = extract_tfidf_export_payload(pipeline)
    # Coef length is vocab + 20 URL features.
    assert int(payload["n_url"]) == len(URL_FEATURE_NAMES)
    # Write sidecars to prove JSON serialization works.
    write_tfidf_sidecars(payload, tmp_path, threshold=0.30)
    # Sidecar files must exist for the frontend copy step.
    assert (tmp_path / "tfidf.json").exists()
    # URL scaler sidecar is required for TypeScript z-scoring.
    assert (tmp_path / "url_scaler.json").exists()
    # Build the tokenizer-free ONNX head.
    onnx_path = tmp_path / "logistic_head.onnx"
    # Gemm weights come from sklearn coef_.
    build_logistic_head_onnx(payload["coef"], [payload["intercept"]], onnx_path)
    # Transform a couple of DMs the same way sklearn does.
    sample = [
        "let's grab lunch tomorrow at noon",
        "urgent: your account will be suspended, verify your password now",
    ]
    # Dense FeatureUnion output is the ORT input (A5 float tensor).
    features = pipeline.named_steps["features"].transform(sample).toarray()
    # sklearn P(scam) is the reference.
    sklearn_p = pipeline.predict_proba(sample)[:, 1]
    # ORT CPU P(scam) must match within float32 noise.
    onnx_p = run_logistic_onnx(onnx_path, features)
    # Allow a little slack for float32 Gemm vs sklearn float64.
    np.testing.assert_allclose(onnx_p, sklearn_p, rtol=1e-4, atol=1e-5)
    # Cross-check the first row with a manual sigmoid too.
    logit = float(np.dot(features[0], np.asarray(payload["coef"])) + payload["intercept"])
    # Manual sigmoid should agree with ORT.
    assert abs(_sigmoid(logit) - float(onnx_p[0])) < 1e-5


# Local sigmoid used by the logistic cross-check.
def _sigmoid(logit: float) -> float:
    """Return a numerically stable logistic sigmoid."""

    # Mirror onnx_export._sigmoid so the test does not import a private helper.
    if logit >= 0:
        # Standard path for non-negative logits.
        return 1.0 / (1.0 + math.exp(-logit))
    # Negative-logit path avoids overflow.
    exp_v = math.exp(logit)
    # Equivalent sigmoid.
    return exp_v / (1.0 + exp_v)


# Unpadded LSTM ONNX logits must match PyTorch on a tiny trained network.
def test_lstm_onnx_matches_unpadded_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert ExportableWordBiLstm ONNX logits match PyTorch for batch=1."""

    # Pin LSTM training to CPU so pytest never needs CUDA.
    cpu = torch.device("cpu")

    # Stub the device helper used by train_model.
    def _cpu_only() -> tuple[torch.device, bool, str]:
        # Report fp32 CPU, matching unit-test convention.
        return cpu, False, "cpu_fp32_lstm_amp_unstable"

    # Patch the production device probe.
    monkeypatch.setattr("secure_chat_ml.lstm.resolve_training_device", _cpu_only)
    # Tiny knobs so the test finishes in well under a second.
    hyperparams = LstmHyperparameters(
        embed_dim=8,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        max_tokens=12,
        max_vocab_size=48,
        batch_size=8,
        eval_batch_size=8,
        learning_rate=1e-2,
        num_train_epochs=1,
        seed=42,
    )
    # TRAIN texts include one URL so the scaler has a non-zero column.
    train_texts = _LEGITIMATE_TEXTS + _SCAM_TEXTS
    # Matching binary labels.
    train_labels = [0] * len(_LEGITIMATE_TEXTS) + [1] * len(_SCAM_TEXTS)
    # Build TRAIN-only vocab.
    token_to_id = build_vocab(train_texts, max_vocab_size=hyperparams.max_vocab_size)
    # Fit the URL scaler on TRAIN only.
    url_scaler = fit_url_scaler(train_texts)
    # Construct the tiny network.
    model = build_model(vocab_size=len(token_to_id), hyperparams=hyperparams)
    # Train one epoch on CPU.
    train_model(
        model,
        train_texts,
        train_labels,
        token_to_id,
        url_scaler,
        hyperparams=hyperparams,
    )
    # Export wrapper must be in eval to drop dropout.
    model.eval()
    # Build the packing-free module.
    exportable = ExportableWordBiLstm(model)
    # Keep the wrapper on CPU eval.
    exportable.cpu().eval()
    # Pick a short scam string that tokenizes to >1 token.
    text = "urgent: your account will be suspended, verify your password now"
    # Encode the way the browser will (no trailing PAD).
    ids = encode_lstm_unpadded(text, token_to_id, max_tokens=hyperparams.max_tokens)
    # Unpadded encode must not be empty.
    assert len(ids) >= 1
    # Tokenizer smoke: the documented splitter returns something for this DM.
    assert tokenize_text(text)
    # Scaled URL row for this one message.
    url_row = transform_url_features([text], url_scaler)
    # PyTorch logits from the exportable module.
    with torch.no_grad():
        # Batch dimension is 1, sequence is the true length.
        torch_logits = exportable(
            torch.tensor([ids], dtype=torch.long),
            torch.from_numpy(url_row),
        )
    # Trace ONNX with a dummy sequence, then run the real ids through ORT.
    dummy_ids = torch.zeros(1, 4, dtype=torch.long)
    # Dummy URL width matches the trained head.
    dummy_url = torch.zeros(1, exportable.url_dim, dtype=torch.float32)
    # Write the graph under tmp_path so reports/ is untouched.
    onnx_path = tmp_path / "lstm.onnx"
    # Dynamic axes match the production exporter.
    torch.onnx.export(
        exportable,
        (dummy_ids, dummy_url),
        str(onnx_path),
        input_names=["token_ids", "url_features"],
        output_names=["logits"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "sequence"},
            "url_features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    # Run ORT CPU on the same unpadded ids.
    import onnxruntime as ort_runtime

    # CPU provider only.
    session = ort_runtime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Token ids as int64 [1, seq].
    ort_ids = np.asarray([ids], dtype=np.int64)
    # URL features as float32 [1, url_dim].
    ort_url = np.asarray(url_row, dtype=np.float32)
    # Fetch logits.
    ort_logits = session.run(["logits"], {"token_ids": ort_ids, "url_features": ort_url})[0]
    # Compare against PyTorch.
    np.testing.assert_allclose(
        np.asarray(ort_logits, dtype=np.float32),
        torch_logits.cpu().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )


# Confirm force_eager_attention actually swaps SDPA modules, not only the config flag.
def test_force_eager_attention_replaces_sdpa_modules(tmp_path: Path) -> None:
    """Assert DistilBERT SDPA layers become eager MultiHeadSelfAttention."""

    # Import the two attention classes so the assertion can use exact types.
    from transformers.models.distilbert.modeling_distilbert import (
        DistilBertSdpaAttention,
        MultiHeadSelfAttention,
    )

    # Minimal vocab so DistilBertTokenizer does not hit the Hub.
    vocab_path = tmp_path / "vocab.txt"
    # One token per line, including the specials DistilBERT expects.
    vocab_path.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n")
    # Local tokenizer; tests must not download distilbert-base-uncased.
    tokenizer = DistilBertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    # Default DistilBertConfig on transformers 4.57 constructs SDPA attention.
    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        hidden_dim=64,
        n_layers=1,
        n_heads=4,
        max_position_embeddings=32,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Random tiny classifier, same construction path as the ONNX match test.
    model = DistilBertForSequenceClassification(config)
    # Precondition: HuggingFace defaulted this build to DistilBertSdpaAttention.
    assert isinstance(model.distilbert.transformer.layer[0].attention, DistilBertSdpaAttention)
    # This is the helper export_distilbert_checkpoint calls before tracing.
    force_eager_attention(model)
    # Read the rewritten attention module after the swap.
    attention = model.distilbert.transformer.layer[0].attention
    # Exact type must be eager MultiHeadSelfAttention, not the SDPA subclass.
    assert type(attention) is MultiHeadSelfAttention
    # Config and modules must agree so DistilBERT.forward does not 4-D-convert the mask.
    assert model.config._attn_implementation == "eager"
    # int64 attention_mask is what the tokenizer and ONNX dummy inputs provide.
    dummy_ids = torch.ones(1, 4, dtype=torch.long)
    # Matching dummy mask of ones (all tokens real).
    dummy_mask = torch.ones(1, 4, dtype=torch.long)
    # Eval so dropout is off.
    model.eval()
    # A long mask must not raise RuntimeError from scaled_dot_product_attention.
    with torch.no_grad():
        # Forward through the swapped eager attention.
        logits = model(input_ids=dummy_ids, attention_mask=dummy_mask).logits
    # Sequence-classification head still emits [batch, 2].
    assert logits.shape == (1, 2)


# Tiny DistilBERT ONNX logits must match PyTorch on a short padded batch.
def test_tiny_distilbert_onnx_matches_pytorch(tmp_path: Path) -> None:
    """Assert DistilBertLogitsWrapper ONNX logits match a local tiny model."""

    # Minimal WordPiece vocab including special tokens.
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        "lunch",
        "urgent",
        "account",
        "verify",
        "now",
    ]
    # Write vocab.txt in the BERT one-token-per-line format.
    vocab_path = tmp_path / "vocab.txt"
    # DistilBertTokenizer reads this file locally (no Hub).
    vocab_path.write_text("\n".join(vocab_tokens) + "\n")
    # Load the tokenizer from the local vocab.
    tokenizer = DistilBertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    # 1-layer 32-dim DistilBERT that exports in a few seconds on CPU.
    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        hidden_dim=64,
        n_layers=1,
        n_heads=4,
        max_position_embeddings=32,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Random weights are enough to compare PyTorch vs ORT.
    model = DistilBertForSequenceClassification(config)
    # Force eager attention so the exporter does not emit SDPA.
    force_eager_attention(model)
    # CPU eval for tracing.
    model.cpu().eval()
    # Wrap so ONNX sees a logits tensor.
    wrapper = DistilBertLogitsWrapper(model)
    # Keep the wrapper in eval.
    wrapper.eval()
    # Write a WordPiece sidecar to prove JSON serialization.
    write_wordpiece_vocab(tokenizer, tmp_path, max_length=16)
    # Sidecar must list the special tokens.
    vocab_doc = json.loads((tmp_path / "wordpiece_vocab.json").read_text())
    # [CLS] must be present for the TypeScript tokenizer.
    assert "[CLS]" in vocab_doc["tokens"]
    # Tokenize one short string the same way the browser will (HF reference).
    encoded = tokenizer(
        "hello lunch",
        truncation=True,
        max_length=16,
        padding="max_length",
        return_tensors="pt",
    )
    # PyTorch logits on the padded batch.
    with torch.no_grad():
        # Same inputs the ONNX graph will receive.
        torch_logits = wrapper(encoded["input_ids"], encoded["attention_mask"])
    # Dummy shorter sequence for tracing; dynamic axes handle padding length 16.
    dummy_ids = torch.ones(1, 4, dtype=torch.long)
    # Dummy mask of ones.
    dummy_mask = torch.ones(1, 4, dtype=torch.long)
    # Write the graph under tmp_path.
    onnx_path = tmp_path / "distilbert.onnx"
    # Trace with dynamic sequence length.
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    # Run ORT CPU on the HF-encoded tensors.
    import onnxruntime as ort_runtime

    # CPU provider only.
    session = ort_runtime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # int64 token ids.
    ort_ids = encoded["input_ids"].cpu().numpy().astype(np.int64)
    # int64 attention mask.
    ort_mask = encoded["attention_mask"].cpu().numpy().astype(np.int64)
    # Fetch logits.
    ort_logits = session.run(
        ["logits"], {"input_ids": ort_ids, "attention_mask": ort_mask}
    )[0]
    # Compare against PyTorch; tiny DistilBERT can still differ at 1e-4.
    np.testing.assert_allclose(
        np.asarray(ort_logits, dtype=np.float32),
        torch_logits.cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
