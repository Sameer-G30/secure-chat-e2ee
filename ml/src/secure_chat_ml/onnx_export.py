"""Export published and sweep-winner checkpoints for ONNX Runtime Web.

A5: TF-IDF vectorization stays outside the graph (vocabulary_/idf_ JSON);
only the logistic head becomes a float-tensor ONNX Gemm+Sigmoid. skl2onnx
TfidfVectorizer conversion is forbidden because ai.onnx.ml Tokenizer ops
are missing from the default ORT Web WASM build.

A6: DistilBERT is exported as its own graph (eager attention, prefer int8).
WordPiece tokenization is a JSON vocab sidecar, not an ONNX node.

Word BiLSTM: ONNX is embedding + LSTM + linear head. Sequences are exported
unpadded (batch=1, dynamic length) so pack_padded_sequence is not required
in the browser. URL features are concatenated in TypeScript after the TRAIN
scaler sidecar is applied.

This module never overwrites reports/*.json and never reads chat_eval for
fitting. A missing published TF-IDF joblib is refit into a separate folder.
"""

# Import json to write sidecars and manifests the browser will fetch.
import json

# Import math for the logistic sigmoid used when scoring TF-IDF fixtures.
import math

# Import shutil to copy tokenizer files and export trees into frontend/public.
import shutil

# Import traceback so one failed DistilBERT export cannot abort the other five.
import traceback

# Import Path for typed checkpoint and export locations.
from pathlib import Path

# Import Any for sklearn/report dictionaries.
from typing import Any

# Import numpy for coefficient matrices, dummy tensors, and fixture vectors.
import numpy as np

# Import onnx helpers to build a tokenizer-free logistic graph.
import onnx

# Import onnxruntime to verify exported graphs and to quantize DistilBERT.
import onnxruntime as ort

# Import torch for DistilBERT/LSTM tracing.
import torch

# Import nn to wrap trained modules in an ONNX-friendly forward.
import torch.nn as nn

# Import joblib to load or dump sklearn pipelines without touching reports.
from joblib import dump, load

# Import numpy_helper to embed Gemm weights as ONNX initializers.
from onnx import TensorProto, helper, numpy_helper

# Import sklearn Pipeline typing used by the TF-IDF export path.
from sklearn.pipeline import Pipeline

# Import the baseline loaders used only when the published joblib is missing.
from secure_chat_ml.baseline import (
    SCAM_LABEL,
    build_pipeline,
    load_processed_corpora,
    stratified_split,
)

# Import DistilBERT load/save helpers; export forces eager attention.
from secure_chat_ml.distilbert import load_saved_classifier as load_distilbert

# Import the trained BiLSTM checkpoint loader and the documented tokenizer.
from secure_chat_ml.lstm import (
    PAD_INDEX,
    UNK_INDEX,
    WordBiLstmClassifier,
    tokenize_text,
)
from secure_chat_ml.lstm import load_saved_classifier as load_lstm

# Import the six-way catalog and the fixed browser fixture DMs.
from secure_chat_ml.onnx_web_catalog import (
    CHECKPOINT_CATALOG,
    ONNX_WEB_FIXTURES,
    resolve_under_ml,
)

# Import URL names so TF-IDF/LSTM sidecars cannot drift from url_features.py.
from secure_chat_ml.url_features import URL_FEATURE_NAMES

# ONNX opset used for Gemm/Sigmoid and the traced PyTorch graphs.
DEFAULT_OPSET = 14

# Name of the logistic-head graph consumed by ORT Web for both TF-IDF sizes.
TFIDF_ONNX_NAME = "logistic_head.onnx"

# Name of the TF-IDF vocabulary/idf sidecar (TypeScript vectorizer input).
TFIDF_VOCAB_NAME = "tfidf.json"

# Shared URL scaler sidecar (mean_/scale_ from TRAIN-fitted StandardScaler).
URL_SCALER_NAME = "url_scaler.json"

# DistilBERT graph filename; int8 is preferred, fp32 is the fallback.
DISTILBERT_ONNX_NAME = "model.onnx"

# DistilBERT fp32 graph kept beside int8 so a WASM abort can switch files.
DISTILBERT_FP32_NAME = "model.fp32.onnx"

# WordPiece token list written for the TypeScript tokenizer.
WORDPIECE_VOCAB_NAME = "wordpiece_vocab.json"

# LSTM graph filename (fp32; the 14 MB checkpoint does not need int8 first).
LSTM_ONNX_NAME = "model.onnx"

# LSTM vocab + scaler + threshold sidecar.
LSTM_META_NAME = "lstm_meta.json"

# Per-checkpoint manifest the browser load-check reads first.
MANIFEST_NAME = "manifest.json"

# Fixture scores written next to the graph so the tab can compare banner on/off.
FIXTURE_SCORES_NAME = "fixture_scores.json"


# Wrap DistilBERT so torch.onnx.export receives a tensor, not a ModelOutput.
class DistilBertLogitsWrapper(nn.Module):
    """Return [batch, 2] logits from input_ids and attention_mask."""

    # Keep the HuggingFace module as a child so parameters export.
    def __init__(self, model: nn.Module) -> None:
        # Register as an nn.Module before assigning children.
        super().__init__()
        # Store the sequence-classification model in eval-friendly form.
        self.model = model

    # Match the ONNX input names used by the TypeScript session.
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return raw logits; softmax and the threshold stay in TypeScript."""

        # Call the HF module with explicit keyword tensors (no token_type_ids).
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # SequenceClassifierOutput.logits is the [batch, 2] tensor we export.
        return outputs.logits


# Wrap a trained word BiLSTM so ONNX does not need pack_padded_sequence.
class ExportableWordBiLstm(nn.Module):
    """Run embedding → BiLSTM → last-state concat → URL concat → logits.

    Callers must pass an unpadded token-id row (true sequence length). That
    matches the browser path (one DM at a time) and equals packed LSTM
    output when there are no PAD tokens in the sequence.
    """

    # Copy embedding/LSTM/head weights from a trained WordBiLstmClassifier.
    def __init__(self, trained: WordBiLstmClassifier) -> None:
        # Register as an nn.Module before assigning children.
        super().__init__()
        # Reuse the trained embedding table, including padding_idx=0.
        self.embedding = trained.embedding
        # Reuse the trained bidirectional LSTM (eval: dropout is unused).
        self.lstm = trained.lstm
        # Reuse the trained linear head that expects pooled ⊕ URL features.
        self.classifier = trained.classifier
        # Remember URL width so dummy inputs match the graph.
        self.url_dim = int(trained.url_dim)
        # Remember hidden size so tests can assert concat width.
        self.hidden_size = int(trained.hidden_size)

    # Export-time forward: no dropout, no packing, dynamic sequence length.
    def forward(self, token_ids: torch.Tensor, url_features: torch.Tensor) -> torch.Tensor:
        """Return [batch, 2] logits from unpadded ids and scaled URL features."""

        # Lookup embeddings for every token in the unpadded sequence.
        embedded = self.embedding(token_ids)
        # Run the BiLSTM on the real tokens only (no PAD contamination).
        _out, (hidden_n, _cell_n) = self.lstm(embedded)
        # Last-layer forward hidden is the end-of-sequence state.
        forward_last = hidden_n[-2]
        # Last-layer backward hidden is the start-of-sequence reverse state.
        backward_last = hidden_n[-1]
        # Pool exactly as WordBiLstmClassifier does at eval.
        pooled = torch.cat([forward_last, backward_last], dim=-1)
        # Match URL dtype to the pooled representation (fp32).
        url_block = url_features.to(dtype=pooled.dtype)
        # Concatenate the TRAIN-scaled URL vector before the linear head.
        combined = torch.cat([pooled, url_block], dim=-1)
        # Project to legitimate/scam logits; softmax stays in TypeScript.
        return self.classifier(combined)


# Force HuggingFace DistilBERT to use the ONNX-friendly eager attention path.
def force_eager_attention(model: nn.Module) -> None:
    """Replace SDPA/flash DistilBERT attention with eager MultiHeadSelfAttention.

    DistilBertForSequenceClassification does not honor set_attn_implementation
    after construction. Setting only config._attn_implementation = "eager"
    desyncs the mask path: the forward skip of SDPA 4-D conversion then feeds
    an int64 2-D mask into scaled_dot_product_attention, which PyTorch rejects.
    Rebuild each layer's attention module so ONNX export traces matmul+softmax.
    """

    # Import attention class names from the same transformers DistilBERT module the model uses.
    from transformers.models.distilbert.modeling_distilbert import DISTILBERT_ATTENTION_CLASSES

    # DistilBERT does not honor set_attn_implementation; calling it only prints a warning.
    # Read the HuggingFace config that TransformerBlock used at construction.
    config = getattr(model, "config", None)
    # DistilBERT-base lives at model.distilbert; a wrapper may not have it.
    distilbert = getattr(model, "distilbert", None)
    # Transformer stacks the per-layer blocks that own attention modules.
    transformer = getattr(distilbert, "transformer", None) if distilbert is not None else None
    # layer is a ModuleList of TransformerBlock.
    layers = getattr(transformer, "layer", None) if transformer is not None else None
    # Nothing to rewrite when this is not a DistilBERT classifier.
    if config is None or layers is None:
        # Still record eager on config when we have one, for a later from_pretrained.
        if config is not None:
            # Keep the flag consistent even if layers were missing.
            config._attn_implementation = "eager"
        # Return so TF-IDF/LSTM callers that pass a dummy module are unharmed.
        return
    # Record eager so DistilBERT.forward does not convert the mask into a 4-D SDPA tensor.
    config._attn_implementation = "eager"
    # MultiHeadSelfAttention is the matmul+softmax implementation ONNX can trace.
    eager_cls = DISTILBERT_ATTENTION_CLASSES["eager"]
    # Rewrite every transformer block; DistilBERT-base has six, the tiny test has one.
    for block in layers:
        # Each TransformerBlock stores its attention as .attention.
        old = getattr(block, "attention", None)
        # Skip a malformed block rather than crashing export.
        if old is None:
            # Continue so a later layer can still be rewritten.
            continue
        # DistilBertSdpaAttention subclasses MultiHeadSelfAttention, so type() must be exact.
        if type(old) is eager_cls:
            # Already eager; do not clone weights.
            continue
        # Build a fresh eager attention with the same dim/n_heads as the trained block.
        replacement = eager_cls(config)
        # Copy q/k/v/out projections so logits stay identical to the trained SDPA module.
        replacement.load_state_dict(old.state_dict())
        # Keep the first parameter as the device/dtype source (CPU export uses CPU).
        reference = next(old.parameters())
        # Move copied weights onto the same device and dtype as the original layer.
        replacement.to(device=reference.device, dtype=reference.dtype)
        # Swap the module so forward and torch.onnx.export both see eager attention.
        block.attention = replacement


# Build a tokenizer-free logistic ONNX graph: Gemm + Sigmoid → P(scam).
def build_logistic_head_onnx(
    coef: np.ndarray,
    intercept: np.ndarray,
    output_path: Path,
    *,
    opset: int = DEFAULT_OPSET,
) -> None:
    """Write an ONNX model that maps [batch, n_features] floats to P(scam).

    Binary sklearn LogisticRegression stores coef_ as (1, n_features) and
    uses a sigmoid. The browser concatenates L2-normalized TF-IDF with the
    TRAIN-scaled URL vector, then feeds that dense row into this graph.
    """

    # Ensure 2-D coef even if a caller passed a 1-D weight vector.
    coef_2d = np.asarray(coef, dtype=np.float32).reshape(1, -1)
    # Intercept is a length-1 vector matching the single scam logit.
    intercept_1d = np.asarray(intercept, dtype=np.float32).reshape(1)
    # Gemm wants W as [n_features, 1] so (X @ W) is [batch, 1].
    weights = coef_2d.T.copy()
    # Count features so the graph's input shape matches the sidecar vocab.
    n_features = int(weights.shape[0])
    # Declare a dynamic-batch float input named 'features'.
    features = helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, [None, n_features]
    )
    # Declare P(scam) as [batch, 1] so ORT Web returns a typed tensor.
    probabilities = helper.make_tensor_value_info(
        "probabilities", TensorProto.FLOAT, [None, 1]
    )
    # Embed W as an initializer rather than a graph input.
    weight_init = numpy_helper.from_array(weights, name="W")
    # Embed the intercept as Gemm's C (bias) initializer.
    bias_init = numpy_helper.from_array(intercept_1d, name="B")
    # Y = X @ W + B, the logistic decision function.
    gemm = helper.make_node("Gemm", ["features", "W", "B"], ["logits"], name="gemm")
    # Convert the decision function into P(scam) with a sigmoid.
    sigmoid = helper.make_node("Sigmoid", ["logits"], ["probabilities"], name="sigmoid")
    # Assemble the two-node graph with the two initializers.
    graph = helper.make_graph(
        [gemm, sigmoid],
        "tfidf_logistic_head",
        [features],
        [probabilities],
        [weight_init, bias_init],
    )
    # Bind the documented opset so ORT Web WASM can load the file.
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )
    # Producer string helps reviewers see this was not skl2onnx+Tokenizer.
    model.producer_name = "secure-chat-ml-a5-logistic-head"
    # Fail here if the graph is malformed rather than in the browser.
    onnx.checker.check_model(model)
    # Ensure the export directory exists before writing bytes.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist the ONNX file the frontend fetches from /ml/<id>/.
    onnx.save(model, str(output_path))


# Load an ONNX graph on CPU ORT and run one dummy feed (export smoke test).
def smoke_onnx_cpu(onnx_path: Path, feeds: dict[str, np.ndarray]) -> None:
    """Fail fast when a just-exported graph cannot run on CPU ORT."""

    # Create a CPU session so DistilBERT int8 failures are caught before the tab.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Run once; the caller supplies dummy tensors matching the graph inputs.
    session.run(None, feeds)


# Run the logistic ONNX graph on a dense feature matrix (CPU ORT).
def run_logistic_onnx(onnx_path: Path, features: np.ndarray) -> np.ndarray:
    """Return P(scam) for each row using the exported Gemm+Sigmoid graph."""

    # Create a CPU session so pytest does not need a GPU or a browser.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # ORT requires float32 with shape [batch, n_features].
    tensor = np.asarray(features, dtype=np.float32)
    # Guard a 1-D caller so a single message still has a batch axis.
    if tensor.ndim == 1:
        # Promote (n_features,) to (1, n_features).
        tensor = tensor.reshape(1, -1)
    # Score the batch; output name is 'probabilities'.
    outputs = session.run(["probabilities"], {"features": tensor})
    # Flatten to a 1-D P(scam) vector aligned with the input rows.
    return np.asarray(outputs[0], dtype=np.float64).reshape(-1)


# Pull TF-IDF, URL scaler, and logistic weights out of a fitted sklearn Pipeline.
def extract_tfidf_export_payload(pipeline: Pipeline) -> dict[str, Any]:
    """Return JSON-safe vocab/idf/scaler/coef dicts from a TRAIN-fitted pipeline."""

    # FeatureUnion lives under the documented 'features' step name.
    features = pipeline.named_steps["features"]
    # transformer_list is the stable FeatureUnion API (no named_transformers_).
    union_steps = dict(features.transformer_list)
    # TfidfVectorizer is the first branch; its columns occupy the left of X.
    vectorizer = union_steps["tfidf"]
    # URL branch is extract → StandardScaler → sparse; scaler is TRAIN-fitted.
    url_pipeline = union_steps["url"]
    # StandardScaler.mean_ / scale_ must travel with the vocab (A5).
    scaler = url_pipeline.named_steps["scale"]
    # LogisticRegression is the named classifier head.
    classifier = pipeline.named_steps["classifier"]
    # Build an index-ordered term list so TypeScript can use array lookup.
    n_vocab = int(len(vectorizer.vocabulary_))
    # Pre-allocate so vocabulary_ insertion order cannot scramble columns.
    terms = [""] * n_vocab
    # Invert token → column index into column → token.
    for term, index in vectorizer.vocabulary_.items():
        # Store the n-gram string (unigram or 'w1 w2' bigram) at its column.
        terms[int(index)] = str(term)
    # idf_ aligns with the same column order as terms.
    idf = np.asarray(vectorizer.idf_, dtype=np.float64).tolist()
    # Binary logistic coef_ is (1, n_vocab + n_url); flatten for JSON.
    coef = np.asarray(classifier.coef_, dtype=np.float64).reshape(-1)
    # intercept_ is length 1 for the binary head.
    intercept = float(np.asarray(classifier.intercept_, dtype=np.float64).reshape(-1)[0])
    # Confirm FeatureUnion width matches coef length before the browser sees it.
    expected = n_vocab + len(URL_FEATURE_NAMES)
    # Fail in Python if a future pipeline drops URL features accidentally.
    if int(coef.shape[0]) != expected:
        # Spell out the mismatch so an operator can inspect the joblib.
        raise ValueError(
            f"logistic coef length {coef.shape[0]} != vocab {n_vocab} + "
            f"URL {len(URL_FEATURE_NAMES)}"
        )
    # Bundle everything TypeScript needs to rebuild the A5 feature vector.
    return {
        "vocabulary_terms": terms,
        "idf": idf,
        "ngram_min": int(vectorizer.ngram_range[0]),
        "ngram_max": int(vectorizer.ngram_range[1]),
        "sublinear_tf": bool(vectorizer.sublinear_tf),
        "use_idf": bool(vectorizer.use_idf),
        "lowercase": bool(vectorizer.lowercase),
        "strip_accents": (
            "unicode" if vectorizer.strip_accents == "unicode" else vectorizer.strip_accents
        ),
        "token_pattern": r"(?u)\b\w\w+\b",
        "norm": "l2",
        "analyzer": "word",
        "n_vocab": n_vocab,
        "n_url": len(URL_FEATURE_NAMES),
        "url_feature_names": list(URL_FEATURE_NAMES),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "C": float(getattr(classifier, "C", 0.0)),
        "live_url_reputation": False,
    }


# Write TF-IDF JSON sidecars (vocab, scaler, logistic) next to the ONNX head.
def write_tfidf_sidecars(
    payload: dict[str, Any],
    output_dir: Path,
    *,
    threshold: float,
) -> None:
    """Persist tfidf.json and url_scaler.json; do not write report metrics."""

    # Ensure the export directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Vocabulary + idf + logistic coef live in one file the TS vectorizer fetches.
    tfidf_doc = {
        "vocabulary_terms": payload["vocabulary_terms"],
        "idf": payload["idf"],
        "ngram_min": payload["ngram_min"],
        "ngram_max": payload["ngram_max"],
        "sublinear_tf": payload["sublinear_tf"],
        "use_idf": payload["use_idf"],
        "lowercase": payload["lowercase"],
        "strip_accents": payload["strip_accents"],
        "token_pattern": payload["token_pattern"],
        "norm": payload["norm"],
        "analyzer": payload["analyzer"],
        "n_vocab": payload["n_vocab"],
        "n_url": payload["n_url"],
        "coef": payload["coef"],
        "intercept": payload["intercept"],
        "C": payload["C"],
        "threshold": float(threshold),
        "live_url_reputation": False,
    }
    # Write the vectorizer sidecar the browser TF-IDF path loads.
    (output_dir / TFIDF_VOCAB_NAME).write_text(json.dumps(tfidf_doc))
    # Write scaler mean/scale separately so LSTM and TF-IDF share one TS helper.
    scaler_doc = {
        "feature_names": payload["url_feature_names"],
        "mean": payload["scaler_mean"],
        "scale": payload["scaler_scale"],
        "live_url_reputation": False,
    }
    # Persist the TRAIN-fitted URL scaler (never a VAL/TEST refit).
    (output_dir / URL_SCALER_NAME).write_text(json.dumps(scaler_doc))


# Logistic sigmoid used to cross-check the ONNX head against sklearn.
def _sigmoid(logit: float) -> float:
    """Return 1 / (1 + exp(-logit)) with a stable negative-logit path."""

    # For large negative logits exp(-x) overflows; invert the fraction.
    if logit >= 0:
        # Standard sigmoid for non-negative logits.
        return 1.0 / (1.0 + math.exp(-logit))
    # exp(logit) is safe when logit is negative.
    exp_v = math.exp(logit)
    # Equivalent to 1/(1+e^{-x}) without overflowing exp(-x).
    return exp_v / (1.0 + exp_v)


# Score fixture DMs with a fitted sklearn pipeline at a frozen threshold.
def score_tfidf_fixtures(pipeline: Pipeline, threshold: float) -> list[dict[str, Any]]:
    """Return per-fixture P(scam) and banner flags from sklearn (not chat_eval)."""

    # Accumulate one JSON object per short DM.
    rows: list[dict[str, Any]] = []
    # Walk the fixed fixture list; never the locked 200-row file.
    for fixture in ONNX_WEB_FIXTURES:
        # Read the DM text the browser will also classify.
        text = str(fixture["text"])
        # sklearn predict_proba[:, scam] is the probability the ONNX sigmoid should match.
        p_scam = float(pipeline.predict_proba([text])[0, SCAM_LABEL])
        # Banner on/off uses the VAL-frozen threshold, never a retune.
        warned = bool(p_scam >= float(threshold))
        # Record gold label only as documentation; the browser does not train on it.
        rows.append(
            {
                "id": fixture["id"],
                "text": text,
                "gold_label": int(fixture["gold_label"]),
                "p_scam": p_scam,
                "warned": warned,
                "threshold": float(threshold),
            }
        )
    # Return the list written into fixture_scores.json.
    return rows


# Fit the published TF-IDF recipe into a separate folder when joblib is missing.
def fit_published_tfidf_pipeline(
    ml_root: Path,
    *,
    C: float,
    max_features: int,
    output_dir: Path,
) -> Pipeline:
    """Fit TRAIN-only with frozen C; write joblib under output_dir, not reports/."""

    # Refuse to treat the locked eval directory as a training source.
    processed_dir = ml_root / "data" / "processed_chat_llm"
    # Load the completed 71k llm_intent_v1 rewrite (do not start the rewriter).
    combined = load_processed_corpora(processed_dir)
    # Reproduce the published 70/20/10 split so TRAIN matches the reports.
    train_df, _val_df, _test_df = stratified_split(combined, random_state=42)
    # Build the published FeatureUnion + logistic head at the frozen C.
    pipeline = build_pipeline(max_features=max_features, C=C)
    # Fit vectorizer, URL scaler, and logistic head on TRAIN only.
    pipeline.fit(train_df["text"], train_df["label"])
    # Park the dump in a dedicated export folder (do not invent reports JSON).
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist joblib so a later re-export can skip the 50k-row fit.
    dump(pipeline, output_dir / "pipeline.joblib")
    # Tell the operator this was an export-only fit.
    print(
        f"fit_published_tfidf_pipeline: wrote {output_dir / 'pipeline.joblib'} "
        "(reports untouched)"
    )
    # Return the live pipeline so the caller can export immediately.
    return pipeline


# Load a TF-IDF joblib, or fit the published recipe into fallback_fit_dir.
def load_or_fit_tfidf_pipeline(ml_root: Path, spec: dict[str, Any]) -> Pipeline:
    """Return a TRAIN-fitted pipeline without writing reports/*.json."""

    # Preferred location: an already-trained joblib from the OFAT sweep or a prior export.
    model_dir = resolve_under_ml(ml_root, str(spec["model_dir"]))
    # Joblib filename is stable across the sweep driver and this export path.
    joblib_path = model_dir / "pipeline.joblib"
    # Use the existing dump when the sweep (or a previous export) already wrote it.
    if joblib_path.exists():
        # joblib unpickles UrlFeatureExtractor from secure_chat_ml.url_features.
        return load(joblib_path)
    # Published 50k baseline never wrote models/baseline/; fit into a separate folder.
    fallback_rel = spec.get("fallback_fit_dir")
    # A sweep winner without a joblib is a hard error (do not silently retrain it).
    if not fallback_rel:
        # Point the operator at the missing dump rather than overwriting reports.
        raise FileNotFoundError(
            f"No pipeline.joblib at {joblib_path}. Re-run the matching training "
            "script into its sweep folder; do not overwrite published reports."
        )
    # Resolve the dedicated export-only model directory.
    fallback_dir = resolve_under_ml(ml_root, str(fallback_rel))
    # Reuse a previous export-only fit when present.
    fallback_joblib = fallback_dir / "pipeline.joblib"
    # Skip the 50k-row fit on the second export run.
    if fallback_joblib.exists():
        # Load the already-fitted published pipeline.
        return load(fallback_joblib)
    # One-off TRAIN fit at the frozen published C / max_features.
    return fit_published_tfidf_pipeline(
        ml_root,
        C=float(spec["C"]),
        max_features=int(spec["max_features"]),
        output_dir=fallback_dir,
    )


# Read TEST + chat-eval numbers from existing reports (never overwrite them).
def read_offline_quality(ml_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Return TEST/chat-eval counts cited beside the new browser measurements."""

    # Resolve the reports folder named in the catalog (published or sweep).
    reports_dir = resolve_under_ml(ml_root, str(spec["reports_dir"]))
    # TF-IDF uses baseline_metrics.json; DistilBERT/LSTM use test_metrics.json.
    metrics_path = reports_dir / str(spec["metrics_name"])
    # Chat-eval JSON is always named chat_style_eval_metrics.json in this repo.
    chat_path = reports_dir / "chat_style_eval_metrics.json"
    # Fail loudly if a catalog path was mistyped; do not invent numbers.
    if not metrics_path.exists():
        # The operator should not retrain just to cite already-published JSON.
        raise FileNotFoundError(f"Missing TEST report {metrics_path}")
    # Parse TEST metrics written by the original training run.
    test_payload = json.loads(metrics_path.read_text())
    # Chat-eval may be missing only in synthetic tests; live export requires it.
    chat_payload = json.loads(chat_path.read_text()) if chat_path.exists() else {}
    # Confusion matrices are [[TN, FP], [FN, TP]] with rows = true labels.
    test_cm = test_payload.get("confusion_matrix", [[0, 0], [0, 0]])
    # Chat-eval uses the same orientation when present.
    chat_cm = chat_payload.get("confusion_matrix", [[0, 0], [0, 0]])
    # TEST accuracy lives on the sklearn classification_report blob.
    test_report = test_payload.get("classification_report", {})
    # Chat-eval accuracy is similarly nested.
    chat_report = chat_payload.get("classification_report", {})
    # Bundle the numbers the README table will sit next to.
    return {
        "reports_dir": str(spec["reports_dir"]),
        "test_accuracy": float(test_report.get("accuracy", 0.0)),
        "test_fn": int(test_cm[1][0]) if test_cm else 0,
        "test_fp": int(test_cm[0][1]) if test_cm else 0,
        "chat_accuracy": float(chat_report.get("accuracy", 0.0)) if chat_report else None,
        "chat_fn": int(chat_cm[1][0]) if chat_cm else None,
        "chat_fp": int(chat_cm[0][1]) if chat_cm else None,
        "note": "Offline TEST/chat-eval only. Browser fixtures are not TEST accuracy.",
    }


# Write the per-checkpoint manifest the browser load-check fetches first.
def write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    """Persist manifest.json describing files, threshold, and offline quality."""

    # Ensure the export directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stable pretty JSON makes code-review diffs readable.
    (output_dir / MANIFEST_NAME).write_text(json.dumps(payload, indent=2))


# Copy an export directory into frontend/public/ml/<id>/ for Vite.
def copy_export_to_frontend(export_dir: Path, frontend_public_ml: Path, dirname: str) -> Path:
    """Replace frontend/public/ml/<dirname> with the freshly exported files."""

    # Destination is gitignored; Vite serves it as /ml/<dirname>/.
    dest = frontend_public_ml / dirname
    # Drop a stale copy so removed files cannot linger (e.g. old fp32 graphs).
    if dest.exists():
        # rmtree is safe here: dest is under frontend/public/ml, not reports/.
        shutil.rmtree(dest)
    # Copy the whole checkpoint tree including ONNX + JSON sidecars.
    shutil.copytree(export_dir, dest)
    # Return the public path so the CLI can print it.
    return dest


# Export one TF-IDF checkpoint (10k winner or 50k published default).
def export_tfidf_checkpoint(
    ml_root: Path,
    spec: dict[str, Any],
    export_root: Path,
    *,
    frontend_public_ml: Path | None = None,
) -> Path:
    """Write logistic_head.onnx + JSON sidecars; do not overwrite reports/."""

    # Load or one-off-fit the sklearn pipeline (reports stay untouched).
    pipeline = load_or_fit_tfidf_pipeline(ml_root, spec)
    # Pull vocab/idf/scaler/coef from the fitted FeatureUnion.
    payload = extract_tfidf_export_payload(pipeline)
    # Resolve this checkpoint's export folder under ml/exports/onnx_web/.
    output_dir = export_root / str(spec["export_dirname"])
    # Create the folder before writing ONNX bytes.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Frozen VAL threshold from the catalog (0.20 winner / 0.30 published).
    threshold = float(spec["threshold"])
    # Write vocabulary + scaler JSON the TypeScript vectorizer fetches.
    write_tfidf_sidecars(payload, output_dir, threshold=threshold)
    # Build the Gemm+Sigmoid head from sklearn coef_/intercept_.
    coef = np.asarray(payload["coef"], dtype=np.float32)
    # Intercept is a scalar in the payload; Gemm wants a length-1 vector.
    intercept = np.asarray([payload["intercept"]], dtype=np.float32)
    # Persist the tokenizer-free ONNX head.
    onnx_path = output_dir / TFIDF_ONNX_NAME
    # Construct the graph with the documented opset.
    build_logistic_head_onnx(coef, intercept, onnx_path)
    # Score the four fixture DMs with sklearn so the tab can compare banners.
    fixture_rows = score_tfidf_fixtures(pipeline, threshold)
    # Write fixture expectations (not TEST accuracy).
    (output_dir / FIXTURE_SCORES_NAME).write_text(json.dumps(fixture_rows, indent=2))
    # Cite existing TEST/chat-eval JSON without rewriting those files.
    offline = read_offline_quality(ml_root, spec)
    # Measure artifact bytes for the browser cost table.
    onnx_bytes = int(onnx_path.stat().st_size)
    # Vocabulary JSON is the bulky TF-IDF artifact (10k vs 50k terms).
    vocab_bytes = int((output_dir / TFIDF_VOCAB_NAME).stat().st_size)
    # Build the manifest the load-check reads to know which files to fetch.
    write_manifest(
        output_dir,
        {
            "id": spec["id"],
            "load_order": spec["load_order"],
            "family": "tfidf",
            "label": spec["label"],
            "threshold": threshold,
            "onnx_file": TFIDF_ONNX_NAME,
            "onnx_bytes": onnx_bytes,
            "quantize": "fp32",
            "sidecars": {
                "tfidf": TFIDF_VOCAB_NAME,
                "url_scaler": URL_SCALER_NAME,
                "fixtures": FIXTURE_SCORES_NAME,
            },
            "artifact_bytes": {
                "onnx": onnx_bytes,
                "tfidf_json": vocab_bytes,
            },
            "offline": offline,
            "wired_in_chatscreen_by_default": spec["id"] == "tfidf_best",
        },
    )
    # Copy into Vite's public folder when the CLI asked for a frontend dest.
    if frontend_public_ml is not None:
        # Replace any previous public copy of this checkpoint id.
        copy_export_to_frontend(output_dir, frontend_public_ml, str(spec["export_dirname"]))
    # Return the export directory for tests and the CLI summary.
    return output_dir


# Encode one DM the same way the browser will (unpadded, truncated, min length 1).
def encode_lstm_unpadded(
    text: str,
    token_to_id: dict[str, int],
    *,
    max_tokens: int,
) -> list[int]:
    """Return token ids with no trailing PAD, matching ExportableWordBiLstm."""

    # Split with the documented whitespace + punctuation tokenizer.
    tokens = tokenize_text(text)
    # Truncate on the right when the DM overflows the training cap.
    if len(tokens) > int(max_tokens):
        # Keep the leading tokens; DistilBERT also truncates the tail.
        tokens = tokens[: int(max_tokens)]
    # Map OOV tokens to UNK using the TRAIN vocabulary.
    unk_id = int(token_to_id.get("<unk>", UNK_INDEX))
    # Convert each token string into an embedding row index.
    ids = [int(token_to_id.get(token, unk_id)) for token in tokens]
    # Empty DMs still need length >= 1 so the LSTM sees one timestep.
    if not ids:
        # A lone PAD token matches encode_texts' empty-message convention.
        return [PAD_INDEX]
    # Return the unpadded id row the ONNX graph consumes.
    return ids


# Export one word-BiLSTM checkpoint (8-epoch winner or 4-epoch published).
def export_lstm_checkpoint(
    ml_root: Path,
    spec: dict[str, Any],
    export_root: Path,
    *,
    frontend_public_ml: Path | None = None,
) -> Path:
    """Write LSTM ONNX + vocab/scaler sidecars; do not overwrite reports/lstm/."""

    # Load the trained word BiLSTM, TRAIN vocab, URL scaler, knobs, threshold.
    model_dir = resolve_under_ml(ml_root, str(spec["model_dir"]))
    # Weights are gitignored; fail clearly if the operator skipped train_lstm.
    model, token_to_id, url_scaler, hyperparams, saved_threshold = load_lstm(model_dir)
    # Catalog threshold is the VAL-frozen cut (0.20 winner / 0.30 published).
    threshold = float(spec["threshold"])
    # Prefer the catalog cut; it must match the checkpoint sidecar.
    if abs(float(saved_threshold) - threshold) > 1e-9:
        # Still export, but warn so a mismatched meta.json is visible.
        print(
            f"export_lstm_checkpoint: {spec['id']} meta threshold "
            f"{saved_threshold} != catalog {threshold}; using catalog"
        )
    # Switch to eval so dropout is a no-op before we copy weights.
    model.eval()
    # Build the packing-free wrapper used by torch.onnx.export.
    exportable = ExportableWordBiLstm(model)
    # Keep the wrapper in eval on CPU for a deterministic graph.
    exportable.cpu().eval()
    # Resolve this checkpoint's export folder.
    output_dir = export_root / str(spec["export_dirname"])
    # Create the folder before writing ONNX bytes.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Dummy unpadded sequence of length 4 is enough to trace the LSTM.
    dummy_ids = torch.zeros(1, 4, dtype=torch.long)
    # Dummy URL block matches the trained head's concat width.
    dummy_url = torch.zeros(1, exportable.url_dim, dtype=torch.float32)
    # Destination path for the fp32 LSTM graph.
    onnx_path = output_dir / LSTM_ONNX_NAME
    # Trace with dynamic batch and sequence axes for one-DM inference.
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
        opset_version=DEFAULT_OPSET,
        dynamo=False,
    )
    # Dummy numpy feeds match the traced LSTM inputs (batch=1, seq=4, url=20).
    lstm_feeds = {
        "token_ids": dummy_ids.numpy(),
        "url_features": dummy_url.numpy(),
    }
    # Fail here if the graph uses ops ORT CPU cannot run (would also fail in WASM).
    smoke_onnx_cpu(onnx_path, lstm_feeds)
    # Write vocab + scaler + knobs so TypeScript can tokenize and scale URLs.
    meta_doc = {
        "architecture": "word_bilstm_url_concat",
        "token_to_id": token_to_id,
        "pad_index": PAD_INDEX,
        "unk_index": UNK_INDEX,
        "max_tokens": int(hyperparams.max_tokens),
        "url_feature_names": list(URL_FEATURE_NAMES),
        "url_dim": int(exportable.url_dim),
        "scaler_mean": np.asarray(url_scaler.mean_, dtype=np.float64).tolist(),
        "scaler_scale": np.asarray(url_scaler.scale_, dtype=np.float64).tolist(),
        "threshold": threshold,
        "embed_dim": int(hyperparams.embed_dim),
        "hidden_size": int(hyperparams.hidden_size),
        "num_layers": int(hyperparams.num_layers),
        "live_url_reputation": False,
        "wired_in_chatscreen": False,
    }
    # Persist the sidecar next to the ONNX graph.
    (output_dir / LSTM_META_NAME).write_text(json.dumps(meta_doc))
    # Also write the shared URL scaler file so TF-IDF and LSTM share one TS loader.
    scaler_doc = {
        "feature_names": list(URL_FEATURE_NAMES),
        "mean": meta_doc["scaler_mean"],
        "scale": meta_doc["scaler_scale"],
        "live_url_reputation": False,
    }
    # Persist scaler JSON used by the TypeScript URL helper.
    (output_dir / URL_SCALER_NAME).write_text(json.dumps(scaler_doc))
    # Score fixtures with the original packed PyTorch model (batch-friendly).
    from secure_chat_ml.lstm import predict_scam_proba

    # Collect texts in catalog order.
    texts = [str(row["text"]) for row in ONNX_WEB_FIXTURES]
    # Frozen PyTorch probabilities (fp32, packed) are the fixture reference.
    py_probs = predict_scam_proba(
        model,
        texts,
        token_to_id,
        url_scaler,
        max_tokens=int(hyperparams.max_tokens),
        batch_size=4,
        device=torch.device("cpu"),
    )
    # Build fixture rows the browser compares against (banner, not TEST acc).
    fixture_rows = []
    # Zip catalog metadata with the PyTorch probability vector.
    for fixture, p_scam in zip(ONNX_WEB_FIXTURES, py_probs, strict=True):
        # Apply the VAL-frozen threshold to decide the banner.
        warned = bool(float(p_scam) >= threshold)
        # Record one fixture object.
        fixture_rows.append(
            {
                "id": fixture["id"],
                "text": fixture["text"],
                "gold_label": int(fixture["gold_label"]),
                "p_scam": float(p_scam),
                "warned": warned,
                "threshold": threshold,
            }
        )
    # Write fixture expectations next to the graph.
    (output_dir / FIXTURE_SCORES_NAME).write_text(json.dumps(fixture_rows, indent=2))
    # Cite existing TEST/chat-eval JSON.
    offline = read_offline_quality(ml_root, spec)
    # Measure ONNX bytes for the cost table.
    onnx_bytes = int(onnx_path.stat().st_size)
    # Write the load-check manifest.
    write_manifest(
        output_dir,
        {
            "id": spec["id"],
            "load_order": spec["load_order"],
            "family": "lstm",
            "label": spec["label"],
            "threshold": threshold,
            "onnx_file": LSTM_ONNX_NAME,
            "onnx_bytes": onnx_bytes,
            "quantize": "fp32",
            "sidecars": {
                "lstm_meta": LSTM_META_NAME,
                "url_scaler": URL_SCALER_NAME,
                "fixtures": FIXTURE_SCORES_NAME,
            },
            "artifact_bytes": {"onnx": onnx_bytes},
            "offline": offline,
            "wired_in_chatscreen_by_default": False,
        },
    )
    # Copy into Vite public/ml when requested.
    if frontend_public_ml is not None:
        # Replace any previous public copy of this LSTM id.
        copy_export_to_frontend(output_dir, frontend_public_ml, str(spec["export_dirname"]))
    # Return the export directory.
    return output_dir


# Dynamically quantize a fp32 ONNX graph to int8 weights (ORT CPU).
def quantize_onnx_dynamic(fp32_path: Path, int8_path: Path) -> None:
    """Write an int8-weight ONNX file beside the fp32 source graph."""

    # Import here so tests that only build Gemm graphs need not touch quantizer internals.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Infer shapes first; some DistilBERT graphs need this before quantize_dynamic.
    inferred = onnx.shape_inference.infer_shapes(onnx.load(str(fp32_path)))
    # Write a temporary inferred model the quantizer can rewrite.
    inferred_path = fp32_path.with_suffix(".inferred.onnx")
    # Persist the shape-inferred fp32 graph next to the export (gitignored).
    onnx.save(inferred, str(inferred_path))
    # Quantize MatMul/Gemm weights to int8; activations stay fp32 (dynamic).
    quantize_dynamic(
        model_input=str(inferred_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    # Drop the inferred intermediate so the export folder stays small.
    inferred_path.unlink(missing_ok=True)


# Dump DistilBERT WordPiece tokens as a JSON array for the TypeScript tokenizer.
def write_wordpiece_vocab(tokenizer: Any, output_dir: Path, *, max_length: int) -> None:
    """Write wordpiece_vocab.json from a HuggingFace DistilBertTokenizer."""

    # HuggingFace get_vocab maps token string → id.
    vocab = tokenizer.get_vocab()
    # Allocate an id-ordered list; missing holes stay empty strings.
    size = int(max(vocab.values()) + 1) if vocab else 0
    # Pre-fill so JSON stays a dense array the browser can index.
    tokens = [""] * size
    # Invert the vocab dict into the array TypeScript will search.
    for token, index in vocab.items():
        # Guard a pathological id that exceeds the allocated length.
        if 0 <= int(index) < size:
            # Store the WordPiece string (including ##subwords).
            tokens[int(index)] = str(token)
    # Bundle special-token names and the truncation length for this checkpoint.
    doc = {
        "tokens": tokens,
        "unk_token": str(tokenizer.unk_token or "[UNK]"),
        "cls_token": str(tokenizer.cls_token or "[CLS]"),
        "sep_token": str(tokenizer.sep_token or "[SEP]"),
        "pad_token": str(tokenizer.pad_token or "[PAD]"),
        "unk_id": int(tokenizer.unk_token_id or 100),
        "cls_id": int(tokenizer.cls_token_id or 101),
        "sep_id": int(tokenizer.sep_token_id or 102),
        "pad_id": int(tokenizer.pad_token_id or 0),
        "do_lower_case": True,
        "max_length": int(max_length),
        "max_input_chars_per_word": 100,
    }
    # Persist the vocab the browser WordPiece implementation loads.
    (output_dir / WORDPIECE_VOCAB_NAME).write_text(json.dumps(doc))


# Export one DistilBERT checkpoint (512-token winner or 256-token Slice 5).
def export_distilbert_checkpoint(
    ml_root: Path,
    spec: dict[str, Any],
    export_root: Path,
    *,
    frontend_public_ml: Path | None = None,
    quantize: bool = True,
) -> Path:
    """Write DistilBERT ONNX (prefer int8) + WordPiece vocab; reports stay intact."""

    # Load the local HuggingFace checkpoint (no Hub fetch; files are on disk).
    model_dir = resolve_under_ml(ml_root, str(spec["model_dir"]))
    # Tokenizer + classification head were saved by train_distilbert.py.
    model, tokenizer = load_distilbert(model_dir)
    # Eager attention avoids SDPA custom ops that ORT Web cannot run.
    force_eager_attention(model)
    # Export from CPU eval; tracing does not need the RTX 4060.
    model.cpu().eval()
    # Wrap so ONNX sees a logits tensor rather than a ModelOutput.
    wrapper = DistilBertLogitsWrapper(model)
    # Keep the wrapper in eval as well.
    wrapper.eval()
    # Catalog max_length is the truncation the browser must apply.
    max_length = int(spec["max_length"])
    # Catalog threshold is the VAL-frozen cut (0.20 winner / 0.30 Slice 5).
    threshold = float(spec["threshold"])
    # Resolve this checkpoint's export folder.
    output_dir = export_root / str(spec["export_dirname"])
    # Create the folder before writing ONNX bytes.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Dummy batch of length 8 is enough to trace; sequence axis is dynamic.
    dummy_len = min(8, max_length)
    # input_ids are int64 token indices.
    dummy_ids = torch.ones(1, dummy_len, dtype=torch.long)
    # attention_mask of ones means every dummy token is real.
    dummy_mask = torch.ones(1, dummy_len, dtype=torch.long)
    # Always keep a fp32 graph so int8 failure can fall back.
    fp32_path = output_dir / DISTILBERT_FP32_NAME
    # Trace DistilBERT with dynamic batch and sequence axes.
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=DEFAULT_OPSET,
        dynamo=False,
    )
    # Default serving file is the int8 graph when quantization succeeds.
    serving_name = DISTILBERT_ONNX_NAME
    # Quantize flag starts true; a quantizer exception leaves fp32 as serving.
    used_quant = "fp32"
    # Try int8; a failure must not abort the rest of the six-way export.
    if quantize:
        # Int8 destination is the filename the browser tries first.
        int8_path = output_dir / DISTILBERT_ONNX_NAME
        try:
            # Dynamic quantization of MatMul/Gemm weights.
            quantize_onnx_dynamic(fp32_path, int8_path)
            # Record that the serving graph is int8.
            used_quant = "int8"
        except Exception as exc:  # noqa: BLE001
            # Keep going with fp32; the load-check will measure whichever file exists.
            print(f"export_distilbert_checkpoint: int8 failed for {spec['id']}: {exc}")
            # Copy fp32 onto the serving filename so the browser has one graph to fetch.
            shutil.copyfile(fp32_path, output_dir / DISTILBERT_ONNX_NAME)
            # Serving graph is fp32 after the failed quantize.
            used_quant = "fp32"
    else:
        # Tests can skip quantization to stay fast.
        shutil.copyfile(fp32_path, output_dir / DISTILBERT_ONNX_NAME)
        # Serving graph is the fp32 copy.
        used_quant = "fp32"
    # Dummy numpy feeds match the traced DistilBERT inputs (int64 ids + mask).
    distilbert_feeds = {
        "input_ids": dummy_ids.numpy(),
        "attention_mask": dummy_mask.numpy(),
    }
    # Serving path is always model.onnx (int8 or the fp32 copy).
    serving_path = output_dir / DISTILBERT_ONNX_NAME
    try:
        # Catch a broken int8 graph before the browser WASM abort.
        smoke_onnx_cpu(serving_path, distilbert_feeds)
    except Exception as exc:  # noqa: BLE001
        # Int8 can fail CPU ORT even when fp32 is fine; fall back to fp32 serving.
        print(f"export_distilbert_checkpoint: CPU smoke failed for {spec['id']}: {exc}")
        # Replace the serving file with the fp32 graph we already traced.
        shutil.copyfile(fp32_path, serving_path)
        # Record that the browser will fetch fp32.
        used_quant = "fp32"
        # If fp32 also fails, abort this checkpoint so the CLI can continue 2–6.
        smoke_onnx_cpu(serving_path, distilbert_feeds)
    # Dump WordPiece tokens for the TypeScript tokenizer.
    write_wordpiece_vocab(tokenizer, output_dir, max_length=max_length)
    # Score fixtures with the original PyTorch model (fp32, not the quantized graph).
    from secure_chat_ml.distilbert import predict_scam_proba

    # Collect fixture texts in catalog order.
    texts = [str(row["text"]) for row in ONNX_WEB_FIXTURES]
    # Frozen PyTorch probabilities are the banner reference (quantization may drift).
    py_probs = predict_scam_proba(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=4,
        device=torch.device("cpu"),
        use_fp16=False,
    )
    # Build fixture rows the browser compares (banner on/off, not exact floats).
    fixture_rows = []
    # Zip catalog metadata with PyTorch P(scam).
    for fixture, p_scam in zip(ONNX_WEB_FIXTURES, py_probs, strict=True):
        # Apply the VAL-frozen threshold.
        warned = bool(float(p_scam) >= threshold)
        # Record one fixture object.
        fixture_rows.append(
            {
                "id": fixture["id"],
                "text": fixture["text"],
                "gold_label": int(fixture["gold_label"]),
                "p_scam": float(p_scam),
                "warned": warned,
                "threshold": threshold,
            }
        )
    # Write fixture expectations next to the graph.
    (output_dir / FIXTURE_SCORES_NAME).write_text(json.dumps(fixture_rows, indent=2))
    # Cite existing TEST/chat-eval JSON.
    offline = read_offline_quality(ml_root, spec)
    # Measure serving ONNX bytes (int8 or fp32).
    serving_path = output_dir / serving_name
    # Stat the file the browser will actually fetch.
    onnx_bytes = int(serving_path.stat().st_size)
    # Write the load-check manifest.
    write_manifest(
        output_dir,
        {
            "id": spec["id"],
            "load_order": spec["load_order"],
            "family": "distilbert",
            "label": spec["label"],
            "threshold": threshold,
            "max_length": max_length,
            "onnx_file": serving_name,
            "onnx_bytes": onnx_bytes,
            "quantize": used_quant,
            "sidecars": {
                "wordpiece": WORDPIECE_VOCAB_NAME,
                "fixtures": FIXTURE_SCORES_NAME,
            },
            "artifact_bytes": {
                "onnx": onnx_bytes,
                "fp32_onnx": int(fp32_path.stat().st_size),
            },
            "offline": offline,
            "wired_in_chatscreen_by_default": False,
        },
    )
    # Copy into Vite public/ml when requested.
    if frontend_public_ml is not None:
        # Replace any previous public copy of this DistilBERT id.
        copy_export_to_frontend(output_dir, frontend_public_ml, str(spec["export_dirname"]))
    # Return the export directory.
    return output_dir


# Dispatch one catalog entry to the matching family exporter.
def export_checkpoint(
    ml_root: Path,
    spec: dict[str, Any],
    export_root: Path,
    *,
    frontend_public_ml: Path | None = None,
    distilbert_quantize: bool = True,
) -> Path:
    """Export one of the six checkpoints; family is read from the catalog."""

    # Branch on the catalog family string.
    family = str(spec["family"])
    # DistilBERT winner and Slice-5 default share one exporter.
    if family == "distilbert":
        # Prefer int8; pytest can disable quantization.
        return export_distilbert_checkpoint(
            ml_root,
            spec,
            export_root,
            frontend_public_ml=frontend_public_ml,
            quantize=distilbert_quantize,
        )
    # Word BiLSTM winner and published default share one exporter.
    if family == "lstm":
        # LSTM stays fp32; the graph is small.
        return export_lstm_checkpoint(
            ml_root,
            spec,
            export_root,
            frontend_public_ml=frontend_public_ml,
        )
    # TF-IDF winner and published default share one exporter.
    if family == "tfidf":
        # Logistic head only; vocab JSON is the TypeScript path (A5).
        return export_tfidf_checkpoint(
            ml_root,
            spec,
            export_root,
            frontend_public_ml=frontend_public_ml,
        )
    # A typo in the catalog should fail before any file is written.
    raise ValueError(f"Unknown checkpoint family: {family}")


# Export every catalog entry in documented load order (1..6).
def export_all_checkpoints(
    ml_root: Path,
    export_root: Path,
    *,
    frontend_public_ml: Path | None = None,
    distilbert_quantize: bool = True,
    only_ids: set[str] | None = None,
) -> list[Path]:
    """Export the six-way set (or a subset) without touching published reports."""

    # Accumulate export directories for the CLI summary.
    written: list[Path] = []
    # Walk the catalog in load_order so logs match the browser sequence.
    ordered = sorted(CHECKPOINT_CATALOG, key=lambda row: int(row["load_order"]))
    # Export each selected checkpoint independently so one failure can continue.
    for spec in ordered:
        # Honor --only so a reviewer can re-export a single graph.
        if only_ids is not None and str(spec["id"]) not in only_ids:
            # Skip this catalog row.
            continue
        # Announce which of the six is running.
        print(f"export_all_checkpoints: [{spec['load_order']}/6] {spec['id']} — {spec['label']}")
        try:
            # Dispatch to the family exporter.
            path = export_checkpoint(
                ml_root,
                spec,
                export_root,
                frontend_public_ml=frontend_public_ml,
                distilbert_quantize=distilbert_quantize,
            )
        except Exception as exc:
            # Browser load-check still runs the remaining ids if one graph is missing.
            print(f"export_all_checkpoints: FAILED {spec['id']}: {exc}")
            # Print the stack so an operator can see OOM vs missing weights.
            traceback.print_exc()
            # Continue 2–6 rather than abandoning the slice on DistilBERT 512.
            continue
        # Remember the directory for the summary list.
        written.append(path)
        # Confirm the folder on disk.
        print(f"export_all_checkpoints: wrote {path}")
    # Return all written directories.
    return written
