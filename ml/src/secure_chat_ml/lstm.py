"""Train and evaluate a word-level Bidirectional LSTM scam classifier.

This module holds testable, importable logic; `scripts/train_lstm.py` is
the thin CLI that points it at the real 71k LLM-rewritten rows.
Unit tests fit a tiny random network on synthetic strings and never load
the 71k corpus, the HuggingFace Hub, or a GPU.

Architecture (word BiLSTM + URL concat):
1. Tokenize with a documented whitespace + punctuation splitter.
2. Build the vocabulary from TRAIN only (UNK for OOV; PAD for batching).
3. Embed tokens from scratch (no GloVe download), run a BiLSTM, and pool
   by concatenating the last forward hidden state with the last backward
   hidden state (via pack_padded_sequence so PAD is not treated as a token).
4. Concatenate the TRAIN-fitted StandardScaler URL feature vector
   (len(URL_FEATURE_NAMES)) onto that pooled text vector BEFORE the
   linear head. Messages with no URL still receive the zero URL vector
   (has_url=0); they are never dropped.
5. Linear head → 2 logits → softmax P(scam). Balanced class weights come
   from TRAIN labels only.

Training protocol (mirrors TF-IDF and DistilBERT):
1. Fit vocab, scaler, embeddings, BiLSTM, and head on TRAIN only.
2. Search only the documented decision-threshold grid on VALIDATION.
3. Selection rule: maximize scam recall subject to legitimate recall >= 0.85.
4. Freeze the threshold, score TEST once, then score the locked chat eval
   set predict-only (never fit or retune on those 200 rows).
5. Do not export ONNX; do not wire the frontend; do not train a char LSTM
   in this module.
"""

# Import json so checkpoints can persist vocab, scaler stats, and knobs as text.
import json

# Import re to split messages into alphanumeric runs and punctuation tokens.
import re

# Import Counter so TRAIN token frequencies can cap the vocabulary.
from collections import Counter

# Import dataclass to bundle hyperparameters and VAL threshold results.
from dataclasses import asdict, dataclass, field, fields

# Import Path for typed checkpoint and report locations.
from pathlib import Path

# Import Any for sklearn classification_report dictionaries.
from typing import Any

# Import numpy for token-id matrices, probability vectors, and scaler arrays.
import numpy as np

# Import pandas so callers can pass Series from the shared corpus loaders.
import pandas as pd

# Import PyTorch for tensors, devices, and the LSTM module.
import torch

# Import nn for Embedding, LSTM, Dropout, Linear, and packed-sequence helpers.
import torch.nn as nn

# Import Dataset/DataLoader so the training loop can minibatch TRAIN rows.
from sklearn.preprocessing import StandardScaler

# Import sklearn's balanced class-weight helper to match DistilBERT / TF-IDF.
from sklearn.utils.class_weight import compute_class_weight

# Import Dataset so DataLoader can iterate encoded rows.
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

# Reuse the baseline split/eval/threshold rule so the three tracks stay comparable.
from secure_chat_ml.baseline import (
    DEFAULT_LEGIT_RECALL_FLOOR,
    DEFAULT_SELECTION_RULE,
    DEFAULT_THRESHOLD_GRID,
    EXPANDED_THRESHOLD_GRID,
    LEGITIMATE_LABEL,
    SCAM_LABEL,
    BaselineEvaluation,
    evaluation_from_predictions,
    pick_operating_point,
    score_threshold_grid,
)

# Reuse on-device URL helpers; never HTTP/DNS/VirusTotal.
from secure_chat_ml.url_features import (
    URL_FEATURE_NAMES,
    extract_message_url_features,
    extract_urls,
    suspicious_url_flag_count,
    url_feature_matrix,
)

# Pad index 0 so nn.Embedding(padding_idx=0) ignores pad positions.
PAD_TOKEN = "<pad>"

# UNK index 1 so TRAIN-OOV and rare tokens share one learned vector.
UNK_TOKEN = "<unk>"

# Integer id reserved for PAD; must stay 0 for padding_idx.
PAD_INDEX = 0

# Integer id reserved for UNK; must stay 1 so encode() can default to it.
UNK_INDEX = 1

# Learned embedding width; not searched on TEST or chat_eval.
DEFAULT_EMBED_DIM = 128

# Hidden size per LSTM direction; bidirectional pooled width is 2x this.
DEFAULT_HIDDEN_SIZE = 128

# Single LSTM layer keeps the model small on an 8 GB GPU and on CPU tests.
DEFAULT_NUM_LAYERS = 1

# Dropout on embeddings and the pooled vector (nn.LSTM dropout needs >1 layer).
DEFAULT_DROPOUT = 0.3

# Pad/truncate to 128 word tokens; DistilBERT used 256 WordPieces for comparison.
DEFAULT_MAX_TOKENS = 128

# Cap TRAIN vocabulary at 25k most-frequent tokens plus PAD/UNK (20k–30k band).
DEFAULT_MAX_VOCAB_SIZE = 25_000

# Train batch; 128 fits easily for this tiny network on the RTX 4060.
DEFAULT_BATCH_SIZE = 128

# Eval batches can be larger because inference has no optimizer state.
DEFAULT_EVAL_BATCH_SIZE = 256

# Adam learning rate for from-scratch embeddings + LSTM (not a pretrained LM).
DEFAULT_LEARNING_RATE = 1e-3

# Four epochs sits in the documented 3–5 range; not searched on TEST/chat_eval.
DEFAULT_NUM_TRAIN_EPOCHS = 4

# Clip recurrent gradients so a rare exploding step cannot NaN the run.
DEFAULT_GRAD_CLIP = 1.0

# Adam L2 penalty; documented default is 0.0 (no decay) so published weights match.
DEFAULT_WEIGHT_DECAY = 0.0

# TRAIN class-weight mode; "balanced" matches DistilBERT / TF-IDF.
DEFAULT_CLASS_WEIGHT = "balanced"

# Concatenate the TRAIN-scaled URL vector unless an OFAT run ablates it.
DEFAULT_URL_FEATURES = True

# OFAT VAL grid: published 0.30..0.70 plus the extra 0.20 and 0.25 cuts.
LSTM_EXPANDED_THRESHOLD_GRID: tuple[float, ...] = EXPANDED_THRESHOLD_GRID

# Checkpoint payload files under ml/models/lstm/ (directory is gitignored).
MODEL_WEIGHTS_NAME = "model.pt"

# Sidecar JSON for vocab, scaler, threshold, and hyperparameters.
MODEL_META_NAME = "meta.json"

# Fraction of message characters that must be URL spans to count as link-heavy.
LINK_HEAVY_URL_CHAR_FRACTION = 0.30

# Extra chat-eval misses vs TF-IDF that trip criterion A (10+ of 100 scams).
CHAR_LSTM_A_CHAT_EXTRA_FN = 10

# TEST scam-recall gap vs DistilBERT that counts as a "clearly large" A miss.
CHAR_LSTM_A_TEST_RECALL_GAP = 0.05

# Gap vs TF-IDF URL-bearing (or overall) scam recall that still looks "material".
CHAR_LSTM_C_RECALL_GAP = 0.05

# Chat-eval scam recall below this is treated as unusable for a third baseline.
CHAR_LSTM_USABLE_CHAT_SCAM_RECALL = 0.80

# TEST scam recall within this of DistilBERT still counts as "near" in-domain.
CHAR_LSTM_NEAR_DISTILBERT_TEST = 0.02

# Re-export the shared VAL selection-rule string so scripts do not retype it.
SELECTION_RULE = DEFAULT_SELECTION_RULE


# Bundle the documented word-BiLSTM hyperparameters (frozen except threshold).
@dataclass(frozen=True)
class LstmHyperparameters:
    """Represent the word-BiLSTM knobs recorded in reports (not searched on TEST)."""

    # Record the learned embedding width.
    embed_dim: int = DEFAULT_EMBED_DIM
    # Record the per-direction LSTM hidden size.
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    # Record how many stacked BiLSTM layers were used.
    num_layers: int = DEFAULT_NUM_LAYERS
    # Record dropout applied to embeddings and the pooled representation.
    dropout: float = DEFAULT_DROPOUT
    # Record the pad/truncate length in word tokens.
    max_tokens: int = DEFAULT_MAX_TOKENS
    # Record the TRAIN vocabulary cap (PAD/UNK are extra reserved slots).
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE
    # Record the training minibatch size.
    batch_size: int = DEFAULT_BATCH_SIZE
    # Record the inference minibatch size.
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    # Record the Adam learning rate.
    learning_rate: float = DEFAULT_LEARNING_RATE
    # Record how many epochs TRAIN was shown (VAL is not used as extra epochs).
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    # Record the gradient-norm clip applied after each backward pass.
    grad_clip: float = DEFAULT_GRAD_CLIP
    # Record Adam weight decay (0.0 keeps the published LSTM recipe).
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    # Record TRAIN class-weight mode ("balanced" or "none").
    class_weight: str = DEFAULT_CLASS_WEIGHT
    # Record whether the scaled URL vector is concatenated before the head.
    url_features: bool = DEFAULT_URL_FEATURES
    # Record the seed applied to torch/numpy and the split.
    seed: int = 42
    # Record the pooling rule so reports do not imply mean/max pooling.
    pooling: str = "last_forward_last_backward"
    # Record the tokenizer so reviewers do not assume WordPiece or whitespace-only.
    tokenizer: str = "whitespace_and_punctuation"


# Bundle VAL threshold selection so TEST and chat-eval reuse one frozen cut.
@dataclass
class LstmThresholdResult:
    """Represent the decision threshold frozen on the validation split only."""

    # Record the selected probability threshold for the scam class.
    threshold: float
    # Record why this threshold was chosen (floor feasible vs F1 fallback).
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
    # Record the thresholds that were searched.
    grid_thresholds: list[float] = field(default_factory=list)


# Hold padded token ids, true lengths, scaled URL features, and labels.
class EncodedLstmDataset(Dataset):
    """Return one padded word-id row plus URL features and a binary label."""

    # Store matrices aligned by construction from encode_texts + url_feature_matrix.
    def __init__(
        self,
        token_ids: np.ndarray,
        lengths: np.ndarray,
        url_features: np.ndarray,
        labels: list[int],
    ) -> None:
        # Keep token ids as int64 for nn.Embedding.
        self.token_ids = np.asarray(token_ids, dtype=np.int64)
        # Keep lengths as int64 for pack_padded_sequence.
        self.lengths = np.asarray(lengths, dtype=np.int64)
        # Keep scaled URL features as float32 to match the linear head.
        self.url_features = np.asarray(url_features, dtype=np.float32)
        # Store labels as a plain list so __getitem__ can box a Python int.
        self.labels = labels

    # Report dataset length so DataLoader knows how many batches to build.
    def __len__(self) -> int:
        # Length follows the label vector, which matches encoding rows.
        return len(self.labels)

    # Return tensors for one example; default collate stacks them into a batch.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Copy the padded token-id row for this message.
        return {
            "token_ids": torch.from_numpy(self.token_ids[index]),
            # Copy the true (unpadded) token count used by packing.
            "lengths": torch.tensor(int(self.lengths[index]), dtype=torch.long),
            # Copy the TRAIN-scaled URL feature vector (zeros when no URL).
            "url_features": torch.from_numpy(self.url_features[index]),
            # Attach the integer class label expected by CrossEntropyLoss.
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


# Word BiLSTM with a linear head that sees LSTM pooled states plus URL features.
class WordBiLstmClassifier(nn.Module):
    """Classify a padded token-id sequence using BiLSTM pooling and URL concat."""

    # Construct embeddings, the bidirectional LSTM, dropout, and the 2-way head.
    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        url_dim: int | None = None,
        pad_index: int = PAD_INDEX,
    ) -> None:
        # Initialize nn.Module so parameters register for Adam.
        super().__init__()
        # Default URL width to the frozen lexical/structural feature count.
        resolved_url_dim = len(URL_FEATURE_NAMES) if url_dim is None else int(url_dim)
        # Store hidden size so tests can assert pooled width without a forward.
        self.hidden_size = int(hidden_size)
        # Store URL width so tests can assert concat size.
        self.url_dim = resolved_url_dim
        # Store pad index so packing/embedding stay consistent with encode_texts.
        self.pad_index = int(pad_index)
        # Learn token vectors from scratch; padding_idx keeps PAD at zero.
        self.embedding = nn.Embedding(
            num_embeddings=int(vocab_size),
            embedding_dim=int(embed_dim),
            padding_idx=self.pad_index,
        )
        # nn.LSTM dropout is a no-op for one layer; we still pass 0.0 then.
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        # Bidirectional LSTM consumes embeddings and emits per-direction hiddens.
        self.lstm = nn.LSTM(
            input_size=int(embed_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        # Drop embedding coordinates before the recurrent core (works at 1 layer).
        self.embed_dropout = nn.Dropout(float(dropout))
        # Drop the pooled text vector before concatenating URL features.
        self.repr_dropout = nn.Dropout(float(dropout))
        # Linear head sees [forward_last; backward_last; scaled_url_features].
        self.classifier = nn.Linear(2 * int(hidden_size) + resolved_url_dim, 2)

    # Expose the concat width so unit tests can assert URL features are included.
    @property
    def combined_dim(self) -> int:
        """Return 2 * hidden_size + url_dim, the classifier input width."""

        # Two directions times hidden, plus the scaled URL block.
        return 2 * self.hidden_size + self.url_dim

    # Run embedding → packed BiLSTM → last-state concat → URL concat → logits.
    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        url_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return [batch, 2] logits from token ids, lengths, and URL features."""

        # Lookup learned embeddings for every (possibly padded) token id.
        embedded = self.embedding(token_ids)
        # Apply dropout to embeddings so the LSTM cannot memorize TRAIN n-grams.
        embedded = self.embed_dropout(embedded)
        # pack_padded_sequence requires lengths on CPU in some PyTorch builds.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        # Run the BiLSTM; packed inputs skip PAD so last hidden is a real token.
        _packed_out, (hidden_n, _cell_n) = self.lstm(packed)
        # hidden_n layout: [layer0_fwd, layer0_bwd, layer1_fwd, layer1_bwd, ...].
        forward_last = hidden_n[-2]
        # Last backward hidden is the start-of-sequence context from the reverse pass.
        backward_last = hidden_n[-1]
        # Pool by concatenating the two directional last states (not mean/max).
        pooled = torch.cat([forward_last, backward_last], dim=-1)
        # Drop the pooled text vector before the head.
        pooled = self.repr_dropout(pooled)
        # Match URL dtype to pooled dtype (fp32 path; LSTM AMP is not used).
        url_block = url_features.to(device=pooled.device, dtype=pooled.dtype)
        # Concatenate scaled URL features so links are not only UNK token spans.
        combined = torch.cat([pooled, url_block], dim=-1)
        # Project the concatenated vector to legitimate/scam logits.
        return self.classifier(combined)


# Refuse to treat the locked eval directory as a training or vocab source.
def assert_not_chat_eval_path(path: Path) -> None:
    """Raise ValueError if a path is inside the locked chat_eval directory."""

    # Inspect every part so nested chat_eval copies cannot sneak into fit().
    if "chat_eval" in Path(path).parts:
        # Honor evaluation_policy.chat_style_eval_training_allowed: false.
        raise ValueError(
            f"Refusing to read or write {path}: the locked chat-style eval set "
            "must stay out of LSTM fitting and threshold search "
            "(chat_style_eval_training_allowed: false)."
        )


# Convert a pandas Series or list into a plain list of strings.
def as_text_list(texts: pd.Series | list[str] | np.ndarray) -> list[str]:
    """Return message texts as `list[str]` for tokenization and URL features."""

    # pandas Series is the usual caller from load_processed_corpora.
    if isinstance(texts, pd.Series):
        # Cast through str so NaN never reaches the tokenizer as float.
        return texts.astype(str).tolist()
    # numpy arrays of object/str also show up in tests.
    if isinstance(texts, np.ndarray):
        # Convert each cell to str without going through pandas.
        return [str(text) for text in texts.tolist()]
    # A Python list is already the tokenizer's preferred input type.
    return [str(text) for text in texts]


# Convert a pandas Series or list into integer labels 0/1.
def as_label_list(labels: pd.Series | list[int] | np.ndarray) -> list[int]:
    """Return binary labels as `list[int]` matching data/label-schema.yaml."""

    # pandas Series is the usual caller from the stratified split.
    if isinstance(labels, pd.Series):
        # Cast to int so boolean/float label columns cannot sneak through.
        return labels.astype(int).tolist()
    # numpy arrays need a Python list for EncodedLstmDataset.
    if isinstance(labels, np.ndarray):
        # Flatten then int-cast each cell.
        return [int(label) for label in labels.tolist()]
    # A Python list is already usable.
    return [int(label) for label in labels]


# Seed numpy, Python, and torch so TRAIN runs are reproducible.
def set_reproducible_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) with the split seed."""

    # Import random locally so the library module does not need it at import time.
    import random

    # Seed Python's RNG used by some shuffling helpers.
    random.seed(seed)
    # Seed NumPy used by class-weight computation and metric helpers.
    np.random.seed(seed)
    # Seed PyTorch CPU generators.
    torch.manual_seed(seed)
    # Seed every CUDA device when a GPU is visible.
    if torch.cuda.is_available():
        # Keep GPU dropout aligned with the CPU seed.
        torch.cuda.manual_seed_all(seed)


# Describe the device; LSTM training stays fp32 even on CUDA.
def resolve_training_device() -> tuple[torch.device, bool, str]:
    """Return (device, use_fp16, reason) for this machine.

    fp16/AMP is not used for this LSTM. Recurrent cells overflow more easily
    under mixed precision than DistilBERT matmuls, and the word BiLSTM is
    small enough that fp32 fits the RTX 4060 8 GB with room to spare.
    CPU tests always stay fp32.
    """

    # Prefer the RTX 4060 when the CUDA driver is visible to this WSL2 process.
    if torch.cuda.is_available():
        # Build a cuda:0 device handle for model.to(...) and inference.
        device = torch.device("cuda")
        # Keep LSTM math in fp32; record why AMP was skipped.
        return device, False, "cuda_fp32_lstm_amp_unstable"
    # Fall back to CPU fp32 when no GPU is present (unit tests, CI).
    device = torch.device("cpu")
    # Document that fp16 was skipped rather than silently implying CUDA AMP.
    return device, False, "cpu_fp32_fp16_requires_cuda"


# Compute sklearn-balanced class weights from TRAIN labels only.
def balanced_class_weights(train_labels: list[int]) -> torch.Tensor:
    """Return a length-2 CPU tensor of balanced weights for CrossEntropyLoss."""

    # Fit weights on TRAIN labels only so VAL/TEST cannot leak into the loss.
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([LEGITIMATE_LABEL, SCAM_LABEL], dtype=int),
        y=np.asarray(train_labels, dtype=int),
    )
    # Return float32 CPU weights; the training loop moves them onto the logits device.
    return torch.tensor(weights, dtype=torch.float32)


# Split a message into lowercase alphanumeric runs and single punctuation tokens.
def tokenize_text(text: str) -> list[str]:
    """Return word tokens using whitespace + punctuation splitting.

    Each `[a-z0-9]+` run is one token. Every other non-whitespace character
    (including `/`, `.`, `:`, `?`) is its own token. Whitespace is dropped.
    URLs therefore explode into many short tokens and OOV hostnames become
    UNK — which is why the scaled URL feature vector is concatenated onto
    the BiLSTM pooled representation rather than relying on token identity.
    """

    # Missing or empty text produces no tokens; encode_texts inserts PAD.
    if not text:
        # Return an empty list so callers can test emptiness uniformly.
        return []
    # Lowercase first so TRAIN/VAL/TEST share one vocabulary case.
    lowered = str(text).lower()
    # Find alphanumeric runs or single non-whitespace, non-alnum characters.
    return re.findall(r"[a-z0-9]+|[^a-z0-9\s]", lowered)


# Build PAD/UNK plus the most frequent TRAIN tokens, capped at max_vocab_size.
def build_vocab(
    train_texts: pd.Series | list[str] | np.ndarray,
    *,
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
) -> dict[str, int]:
    """Return token→id from TRAIN texts only; VAL/TEST/chat_eval never enter."""

    # Materialize TRAIN strings so a Series and a list share one counting path.
    text_list = as_text_list(train_texts)
    # Count token occurrences across TRAIN only.
    counts: Counter[str] = Counter()
    # Walk every TRAIN message.
    for text in text_list:
        # Update frequencies with the documented whitespace/punctuation tokens.
        counts.update(tokenize_text(text))
    # Reserve two slots so the cap is "most-frequent tokens" plus PAD/UNK.
    reserved = 2
    # Keep at least PAD/UNK even if the operator passes a tiny cap in tests.
    usable_cap = max(int(max_vocab_size) - reserved, 0)
    # Take the most frequent TRAIN tokens until the cap is hit.
    most_common = counts.most_common(usable_cap)
    # Start the map with the reserved special tokens at stable indices.
    token_to_id: dict[str, int] = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    # Assign ids in frequency order so reports can dump a stable vocab size.
    for token, _count in most_common:
        # Skip accidental collisions with the reserved names.
        if token in token_to_id:
            # Do not overwrite PAD/UNK if a corpus somehow contained those strings.
            continue
        # Append the next integer id.
        token_to_id[token] = len(token_to_id)
    # Return the TRAIN-only vocabulary used by encode_texts.
    return token_to_id


# Count how many texts would exceed max_tokens without truncation.
def count_truncated_texts(
    texts: pd.Series | list[str] | np.ndarray,
    max_tokens: int,
) -> int:
    """Return how many messages overflow `max_tokens` word tokens."""

    # Materialize texts so pandas/list/ndarray callers share one path.
    text_list = as_text_list(texts)
    # Count rows whose token length is strictly greater than the training cap.
    return sum(len(tokenize_text(text)) > int(max_tokens) for text in text_list)


# Convert texts into a padded token-id matrix plus true lengths.
def encode_texts(
    texts: pd.Series | list[str] | np.ndarray,
    token_to_id: dict[str, int],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (token_ids [n, max_tokens], lengths [n]) using TRAIN vocab only.

    Tokens absent from `token_to_id` become UNK. Sequences longer than
    `max_tokens` are truncated on the right. Empty messages become a single
    PAD token with length 1 so pack_padded_sequence never sees length 0.
    """

    # Materialize texts as a list so we can allocate the padded matrix.
    text_list = as_text_list(texts)
    # Allocate the padded id matrix filled with PAD.
    token_ids = np.full((len(text_list), int(max_tokens)), PAD_INDEX, dtype=np.int64)
    # Allocate the true-length vector (at least 1 per row).
    lengths = np.ones(len(text_list), dtype=np.int64)
    # Look up UNK once so OOV tokens share one id.
    unk_id = int(token_to_id.get(UNK_TOKEN, UNK_INDEX))
    # Encode every message independently.
    for row_index, text in enumerate(text_list):
        # Split with the documented tokenizer.
        tokens = tokenize_text(text)
        # Truncate on the right when the DM overflows max_tokens.
        if len(tokens) > int(max_tokens):
            # Keep the leading tokens; DistilBERT also truncates the tail.
            tokens = tokens[: int(max_tokens)]
        # Map each token through the TRAIN vocab, defaulting to UNK.
        ids = [int(token_to_id.get(token, unk_id)) for token in tokens]
        # Empty DMs still need length >= 1 for packing.
        if not ids:
            # A lone PAD token is ignored by packing/embedding padding_idx.
            ids = [PAD_INDEX]
        # Write ids into the padded row.
        token_ids[row_index, : len(ids)] = np.asarray(ids, dtype=np.int64)
        # Store the unpadded length used by pack_padded_sequence.
        lengths[row_index] = len(ids)
    # Return matrices ready for EncodedLstmDataset.
    return token_ids, lengths


# Fit StandardScaler on TRAIN URL features only (same idea as TF-IDF FeatureUnion).
def fit_url_scaler(
    train_texts: pd.Series | list[str] | np.ndarray,
) -> StandardScaler:
    """Return a StandardScaler fitted on TRAIN URL rows; never VAL/TEST/chat_eval."""

    # Build the dense lexical/structural matrix with zeros for link-free DMs.
    train_matrix = url_feature_matrix(as_text_list(train_texts))
    # Fit mean/scale on TRAIN only so later splits cannot leak into z-scores.
    scaler = StandardScaler()
    # sklearn StandardScaler.fit stores mean_ and scale_ for transform().
    scaler.fit(train_matrix)
    # Return the fitted scaler the VAL/TEST/chat_eval paths must reuse.
    return scaler


# Transform texts with a TRAIN-fitted URL scaler (never refit).
def transform_url_features(
    texts: pd.Series | list[str] | np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Return scaled URL features; all-zero raw rows stay valid (has_url=0)."""

    # Extract the raw lexical/structural matrix, including all-zero ham rows.
    raw = url_feature_matrix(as_text_list(texts))
    # Apply TRAIN mean/scale; do not call fit here.
    scaled = scaler.transform(raw)
    # Cast to float32 for the linear head.
    return np.asarray(scaled, dtype=np.float32)


# Reconstruct a fitted StandardScaler from JSON-safe mean/scale arrays.
def scaler_from_arrays(
    mean: list[float] | np.ndarray,
    scale: list[float] | np.ndarray,
    *,
    var: list[float] | np.ndarray | None = None,
) -> StandardScaler:
    """Return a StandardScaler with TRAIN statistics restored from a checkpoint."""

    # Build an unfitted scaler and stamp the learned statistics onto it.
    scaler = StandardScaler()
    # Restore the TRAIN feature-wise mean.
    scaler.mean_ = np.asarray(mean, dtype=np.float64)
    # Restore the TRAIN feature-wise scale (std, with 0-std columns set to 1).
    scaler.scale_ = np.asarray(scale, dtype=np.float64)
    # Restore variance when the checkpoint stored it; else square the scale.
    scaler.var_ = (
        np.asarray(var, dtype=np.float64)
        if var is not None
        else np.square(scaler.scale_)
    )
    # sklearn 1.7 check_is_fitted looks for n_features_in_.
    scaler.n_features_in_ = int(scaler.mean_.shape[0])
    # Return the reconstructed scaler for transform_url_features.
    return scaler


# Build an untrained WordBiLstmClassifier from hyperparameters and vocab size.
def build_model(
    vocab_size: int,
    hyperparams: LstmHyperparameters | None = None,
) -> WordBiLstmClassifier:
    """Return an unfitted word BiLSTM whose head width includes URL features."""

    # Fall back to documented defaults when the CLI did not override knobs.
    knobs = hyperparams or LstmHyperparameters()
    # Use the frozen URL width, or 0 when an OFAT run ablates URL concat.
    url_dim = len(URL_FEATURE_NAMES) if knobs.url_features else 0
    # Construct the network; the linear head width follows url_dim.
    return WordBiLstmClassifier(
        vocab_size=int(vocab_size),
        embed_dim=knobs.embed_dim,
        hidden_size=knobs.hidden_size,
        num_layers=knobs.num_layers,
        dropout=knobs.dropout,
        url_dim=url_dim,
        pad_index=PAD_INDEX,
    )


# Build the URL block the BiLSTM head expects (empty when url_dim is 0).
def url_feature_array_for_model(
    texts: pd.Series | list[str] | np.ndarray,
    url_scaler: StandardScaler,
    url_dim: int,
) -> np.ndarray:
    """Return scaled URL rows, or a (n, 0) matrix when URL concat is ablated."""

    # Count rows so an empty URL block still has the right first dimension.
    n_rows = len(as_text_list(texts))
    # url_dim 0 means the linear head has no URL columns.
    if int(url_dim) <= 0:
        # Shape (n, 0) concatenates as a no-op on the feature axis.
        return np.zeros((n_rows, 0), dtype=np.float32)
    # Otherwise reuse the TRAIN-fitted scaler (never refit here).
    return transform_url_features(texts, url_scaler)


# Fit embeddings, BiLSTM, and the classification head on TRAIN only.
def train_model(
    model: WordBiLstmClassifier,
    train_texts: pd.Series | list[str],
    train_labels: pd.Series | list[int],
    token_to_id: dict[str, int],
    url_scaler: StandardScaler,
    *,
    hyperparams: LstmHyperparameters | None = None,
    class_weights: torch.Tensor | None = None,
) -> WordBiLstmClassifier:
    """Fit the word BiLSTM on TRAIN only and return the trained model (in place)."""

    # Fall back to documented defaults when the CLI did not override knobs.
    hyperparams = hyperparams or LstmHyperparameters()
    # Seed every RNG before shuffling or dropout runs.
    set_reproducible_seed(hyperparams.seed)
    # Resolve CUDA vs CPU; fp16 is always False for this LSTM.
    device, _use_fp16, _fp16_reason = resolve_training_device()
    # Materialize texts as a list for encoding.
    text_list = as_text_list(train_texts)
    # Materialize labels as ints matching the schema.
    label_list = as_label_list(train_labels)
    # Compute TRAIN class weights unless the caller passed an explicit tensor.
    if class_weights is None:
        # "none" matches TF-IDF's class_weight=None OFAT ablation.
        if hyperparams.class_weight == "none":
            # Equal weights leave the empirical TRAIN prior unadjusted.
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        else:
            # Fit balanced weights here so tests can also pass an explicit tensor.
            class_weights = balanced_class_weights(label_list)
    # Encode TRAIN with the TRAIN vocabulary (already built by the caller).
    token_ids, lengths = encode_texts(
        text_list, token_to_id, max_tokens=hyperparams.max_tokens
    )
    # Scale TRAIN URL features, or pass an empty block when URL concat is off.
    url_features = url_feature_array_for_model(text_list, url_scaler, model.url_dim)
    # Wrap encodings as a Dataset DataLoader can iterate.
    train_dataset = EncodedLstmDataset(token_ids, lengths, url_features, label_list)
    # Shuffle TRAIN each epoch; pin_memory only helps when CUDA is in use.
    loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    # Move the network onto the training device before the first batch.
    model.to(device)
    # Switch to train so dropout is active.
    model.train()
    # Adam on all parameters; lr and weight_decay come from documented knobs.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
    )
    # Move TRAIN-only class weights onto the same device as the logits.
    weight = class_weights.to(device=device, dtype=torch.float32)
    # Weighted CE matches DistilBERT WeightedTrainer / sklearn class_weight=balanced.
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    # Repeat the TRAIN set for the documented epoch count (VAL is not extra epochs).
    for epoch in range(hyperparams.num_train_epochs):
        # Accumulate loss for a per-epoch log line.
        running = 0.0
        # Count batches so the mean loss is well-defined even for tiny tests.
        n_batches = 0
        # Walk every TRAIN minibatch.
        for batch in loader:
            # Move token ids onto the training device.
            batch_ids = batch["token_ids"].to(device)
            # Lengths stay a 1-d long tensor; packing will copy them to CPU.
            batch_lengths = batch["lengths"].to(device)
            # Move scaled URL features onto the training device.
            batch_urls = batch["url_features"].to(device)
            # Move labels onto the training device for CE.
            batch_labels = batch["labels"].to(device)
            # Clear previous gradients before this backward pass.
            optimizer.zero_grad(set_to_none=True)
            # Forward pass yields [batch, 2] logits.
            logits = model(batch_ids, batch_lengths, batch_urls)
            # Compute TRAIN-weighted cross-entropy.
            loss = loss_fn(logits, batch_labels)
            # Backpropagate through the BiLSTM and embedding table.
            loss.backward()
            # Clip recurrent gradients to keep fp32 LSTM steps stable.
            nn.utils.clip_grad_norm_(model.parameters(), hyperparams.grad_clip)
            # Apply the Adam update.
            optimizer.step()
            # Track the detached scalar loss for logging.
            running += float(loss.detach().cpu())
            # Count this batch.
            n_batches += 1
        # Mean loss over batches so epoch logs are comparable across batch sizes.
        mean_loss = running / max(n_batches, 1)
        # Print progress so a 50k-row run is auditable from the terminal.
        print(
            f"train_model: epoch {epoch + 1}/{hyperparams.num_train_epochs}  "
            f"mean_loss={mean_loss:.4f}"
        )
    # Switch to eval so later predict_scam_proba does not apply dropout.
    model.eval()
    # Return the same model object now holding trained weights.
    return model


# Score scam-class softmax probabilities without updating weights.
def predict_scam_proba(
    model: WordBiLstmClassifier,
    texts: pd.Series | list[str],
    token_to_id: dict[str, int],
    url_scaler: StandardScaler,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return P(scam) for each text using a frozen word BiLSTM classifier."""

    # Resolve device the same way training did when the caller omitted it.
    resolved_device, _fp16, _reason = resolve_training_device()
    # Honor an explicit device (tests pin CPU even if CUDA exists).
    device = device if device is not None else resolved_device
    # Materialize texts as a list so we can slice batches.
    text_list = as_text_list(texts)
    # Return an empty float array when there is nothing to score.
    if not text_list:
        # Keep dtype float64 to match numpy's default softmax promotion.
        return np.array([], dtype=np.float64)
    # Encode with the TRAIN vocabulary; OOV becomes UNK.
    token_ids, lengths = encode_texts(text_list, token_to_id, max_tokens=max_tokens)
    # Scale URL features, or pass an empty block when the head has url_dim 0.
    url_features = url_feature_array_for_model(text_list, url_scaler, model.url_dim)
    # Move the model onto the inference device and disable dropout.
    model.to(device)
    # Ensure dropout is off; this function must never train.
    model.eval()
    # Accumulate per-batch scam probabilities on CPU.
    batches: list[np.ndarray] = []
    # Disable autograd for the entire scoring loop.
    with torch.no_grad():
        # Walk the texts in eval-sized batches.
        for start in range(0, len(text_list), batch_size):
            # Exclusive end index for this slice.
            end = start + batch_size
            # Token ids for this batch as a long tensor on the inference device.
            batch_ids = torch.from_numpy(token_ids[start:end]).to(device)
            # True lengths for packing.
            batch_lengths = torch.from_numpy(lengths[start:end]).to(device)
            # Scaled URL block for this batch.
            batch_urls = torch.from_numpy(url_features[start:end]).to(device)
            # Forward pass yields [batch, 2] logits in fp32.
            logits = model(batch_ids, batch_lengths, batch_urls)
            # Convert logits to P(scam) = softmax[:, 1].
            batch_probs = torch.softmax(logits, dim=-1)[:, SCAM_LABEL]
            # Move probabilities to CPU numpy for sklearn metrics.
            batches.append(batch_probs.cpu().numpy())
    # Concatenate batches into one probability vector aligned with `texts`.
    return np.concatenate(batches, axis=0)


# Freeze a decision threshold on VAL probabilities using the shared selection rule.
def tune_threshold_on_validation(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    legit_recall_floor: float = DEFAULT_LEGIT_RECALL_FLOOR,
) -> LstmThresholdResult:
    """Return the VAL-frozen threshold; never looks at TEST or chat_eval.

    This helper only accepts already-computed labels and probabilities. It
    has no path argument, so the locked chat-eval CSV cannot be used to
    search a cut.
    """

    # Cast labels to a 1-d numpy array for sklearn.
    labels = np.asarray(as_label_list(y_true), dtype=int)
    # Score every documented threshold; C=0 because this LSTM has no C grid.
    candidates = score_threshold_grid(labels, y_proba, threshold_grid, C=0.0)
    # Apply the same floor-then-F1-fallback rule as TF-IDF and DistilBERT.
    chosen, selection_reason, floor_feasible = pick_operating_point(
        candidates, legit_recall_floor=legit_recall_floor
    )
    # Bundle the frozen cut plus the VAL metrics at that operating point.
    return LstmThresholdResult(
        threshold=float(chosen["threshold"]),
        selection_reason=selection_reason,
        selection_rule=(
            f"maximize scam recall subject to legitimate recall >= {legit_recall_floor:.2f}"
        ),
        legit_recall_floor=float(legit_recall_floor),
        floor_feasible=floor_feasible,
        classification_report=chosen["report"],
        confusion_matrix=chosen["matrix"],
        val_rows=int(len(labels)),
        grid_thresholds=[float(value) for value in threshold_grid],
    )


# Turn scam probabilities plus a frozen threshold into a BaselineEvaluation.
def evaluate_from_proba(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float,
    train_rows: int,
    source_frame: pd.DataFrame | None = None,
    val_rows: int = 0,
) -> BaselineEvaluation:
    """Apply `threshold` to P(scam) and report precision/recall/F1/confusion."""

    # Convert probabilities into hard labels at the frozen operating point.
    y_pred = (np.asarray(y_proba) >= float(threshold)).astype(int)
    # Reuse the baseline metrics bundle so JSON keys match TF-IDF reports.
    return evaluation_from_predictions(
        pd.Series(as_label_list(y_true)),
        y_pred,
        train_rows=train_rows,
        source_frame=source_frame,
        threshold=threshold,
        C=0.0,
        val_rows=val_rows,
    )


# Persist weights + TRAIN vocab/scaler/threshold for later predict-only scoring.
def save_classifier(
    model: WordBiLstmClassifier,
    token_to_id: dict[str, int],
    url_scaler: StandardScaler,
    output_dir: Path,
    *,
    hyperparams: LstmHyperparameters,
    threshold: float,
) -> None:
    """Write model.pt and meta.json under output_dir (gitignored)."""

    # Ensure the checkpoint directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save only the state_dict so torch.load can use weights_only=True.
    torch.save(model.state_dict(), output_dir / MODEL_WEIGHTS_NAME)
    # Serialize TRAIN vocab, scaler stats, knobs, and the VAL-frozen threshold.
    meta = {
        "architecture": "word_bilstm_url_concat",
        "token_to_id": token_to_id,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "pad_index": PAD_INDEX,
        "unk_index": UNK_INDEX,
        "vocab_size": len(token_to_id),
        "url_feature_names": list(URL_FEATURE_NAMES),
        "scaler_mean": np.asarray(url_scaler.mean_, dtype=np.float64).tolist(),
        "scaler_scale": np.asarray(url_scaler.scale_, dtype=np.float64).tolist(),
        "scaler_var": np.asarray(url_scaler.var_, dtype=np.float64).tolist(),
        "hyperparameters": asdict(hyperparams),
        "chosen_threshold": float(threshold),
        "url_features": True,
        "live_url_reputation": False,
        "onnx_exported": False,
        "frontend_wired": False,
    }
    # Write the sidecar JSON next to the weight file.
    (output_dir / MODEL_META_NAME).write_text(json.dumps(meta, indent=2))


# Reload a locally saved word BiLSTM without touching the Hub or the 71k corpus.
def load_saved_classifier(
    model_dir: Path,
) -> tuple[WordBiLstmClassifier, dict[str, int], StandardScaler, LstmHyperparameters, float]:
    """Load a checkpoint written by `save_classifier` from disk only."""

    # Resolve the weight and sidecar paths.
    weights_path = model_dir / MODEL_WEIGHTS_NAME
    # Resolve the metadata sidecar that holds vocab and scaler statistics.
    meta_path = model_dir / MODEL_META_NAME
    # Fail loudly when the training script has not been run yet.
    if not weights_path.exists() or not meta_path.exists():
        # Point the operator at the named training script, not pytest.
        raise FileNotFoundError(
            f"No word-BiLSTM checkpoint at {model_dir}. "
            "Run scripts/train_lstm.py first (not pytest)."
        )
    # Parse the sidecar; this is a trusted local file written by this project.
    meta = json.loads(meta_path.read_text())
    # Rebuild hyperparameters from the recorded dict, ignoring unknown keys.
    allowed = {item.name for item in fields(LstmHyperparameters)}
    # Drop keys a newer checkpoint might add that this code does not know yet.
    hp_payload = {
        key: value
        for key, value in meta["hyperparameters"].items()
        if key in allowed
    }
    # Missing new fields (url_features, weight_decay, ...) use dataclass defaults.
    hyperparams = LstmHyperparameters(**hp_payload)
    # Rebuild the TRAIN vocabulary (token strings → ids).
    token_to_id = {str(key): int(value) for key, value in meta["token_to_id"].items()}
    # Rebuild the TRAIN-fitted URL scaler from stored mean/scale/var.
    url_scaler = scaler_from_arrays(
        meta["scaler_mean"],
        meta["scaler_scale"],
        var=meta.get("scaler_var"),
    )
    # Construct a matching network before loading weights.
    model = build_model(vocab_size=len(token_to_id), hyperparams=hyperparams)
    # Load weights with weights_only=True (state_dict tensors only).
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Copy tensors into the freshly constructed module.
    model.load_state_dict(state)
    # Disable dropout for inference.
    model.eval()
    # Read the VAL-frozen threshold stored next to the weights.
    threshold = float(meta["chosen_threshold"])
    # Return everything the predict-only eval script needs.
    return model, token_to_id, url_scaler, hyperparams, threshold


# Decide whether a message is dominated by URL characters rather than prose.
def is_link_heavy(text: str, features: dict[str, float] | None = None) -> bool:
    """Return True when the DM contains a URL and URLs dominate the characters.

    A message is link-heavy when has_url=1 and the concatenated extracted URL
    strings occupy at least 30% of the stripped message. Ordinary "see this
    doc https://example.com/..." ham is not link-heavy under that rule. The
    extractor can record both `https://host/path` and a schemeless `host/path`
    as distinct strings, so url_count>=2 is not used as a shortcut.
    """

    # Extract lexical URL features when the caller did not already score them.
    scored = features if features is not None else extract_message_url_features(text)
    # Link-free DMs cannot be link-heavy.
    if float(scored.get("has_url", 0.0)) < 1.0:
        # No URL → not link-heavy.
        return False
    # Collect the URL strings already found by the on-device extractor.
    urls = extract_urls(text)
    # Guard the empty-URL case (should not happen when has_url=1).
    if not urls:
        # No spans → not link-heavy.
        return False
    # Measure how much of the message is literally the URL characters.
    url_chars = sum(len(url) for url in urls)
    # Avoid divide-by-zero on whitespace-only text.
    text_len = max(len(str(text).strip()), 1)
    # Flag messages whose body is mostly a URL (including duplicate https/schemeless spans).
    return (url_chars / text_len) >= LINK_HEAVY_URL_CHAR_FRACTION


# Build one FN/FP audit row with on-device URL flags (no network I/O).
def error_record(
    text: str,
    *,
    y_true: int,
    y_pred: int,
    p_scam: float,
    row_index: int,
) -> dict[str, Any]:
    """Return URL-flag fields for one missed scam or warned ham row."""

    # Score the message with the same lexical extractor the model concatenated.
    features = extract_message_url_features(text)
    # Classify the error type from the schema labels.
    error_type = "FN" if int(y_true) == SCAM_LABEL and int(y_pred) == LEGITIMATE_LABEL else "FP"
    # Keep a short preview so the JSON is auditable without dumping full DMs.
    preview = str(text)[:160]
    # Bundle the flags the char-LSTM decision needs.
    return {
        "row_index": int(row_index),
        "error_type": error_type,
        "y_true": int(y_true),
        "y_pred": int(y_pred),
        "p_scam": float(p_scam),
        "text_preview": preview,
        "has_url": int(features["has_url"]),
        "url_count": float(features["url_count"]),
        "is_known_shortener": int(features["is_known_shortener"]),
        "suspicious_tld": int(features["suspicious_tld"]),
        "host_is_ip": int(features["host_is_ip"]),
        "path_has_login_verify_update_password_keywords": int(
            features["path_has_login_verify_update_password_keywords"]
        ),
        "suspicious_url_flag_count": suspicious_url_flag_count(features),
        "link_heavy": bool(is_link_heavy(text, features)),
    }


# Summarize scam recall on URL-bearing vs no-URL slices after a frozen threshold.
def summarize_url_slices(
    texts: pd.Series | list[str],
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Return overall / URL / no-URL scam recall plus FN counts by has_url."""

    # Materialize texts so URL flags align with labels row-wise.
    text_list = as_text_list(texts)
    # Cast labels to a 1-d integer array.
    labels = np.asarray(as_label_list(y_true), dtype=int)
    # Cast predictions to a 1-d integer array.
    preds = np.asarray(y_pred, dtype=int)
    # Score has_url for every row using the existing extractor.
    has_url = np.array(
        [extract_message_url_features(text)["has_url"] >= 1.0 for text in text_list],
        dtype=bool,
    )
    # Helper: scam recall on an optional boolean mask (None = all rows).
    def _scam_recall(mask: np.ndarray | None) -> float | None:
        # Choose all rows or the caller-provided slice.
        choose = np.ones(len(labels), dtype=bool) if mask is None else mask
        # Restrict to gold scams inside the slice.
        gold_scam = choose & (labels == SCAM_LABEL)
        # No gold scams in the slice → recall is undefined, not zero.
        if int(gold_scam.sum()) == 0:
            # Signal "no support" so reports do not fake a 0.0 recall.
            return None
        # Recall = TP / (TP + FN) among gold scams in the slice.
        return float((preds[gold_scam] == SCAM_LABEL).mean())

    # False negatives are gold scams predicted legitimate.
    fn_mask = (labels == SCAM_LABEL) & (preds == LEGITIMATE_LABEL)
    # Bundle slice metrics for TEST and chat-eval JSON.
    return {
        "n_rows": int(len(labels)),
        "n_scam": int((labels == SCAM_LABEL).sum()),
        "n_url": int(has_url.sum()),
        "n_no_url": int((~has_url).sum()),
        "n_url_scam": int((has_url & (labels == SCAM_LABEL)).sum()),
        "n_no_url_scam": int((~has_url & (labels == SCAM_LABEL)).sum()),
        "scam_recall_overall": _scam_recall(None),
        "scam_recall_url": _scam_recall(has_url),
        "scam_recall_no_url": _scam_recall(~has_url),
        "fn_total": int(fn_mask.sum()),
        "fn_url": int((fn_mask & has_url).sum()),
        "fn_no_url": int((fn_mask & ~has_url).sum()),
        "fp_total": int(((labels == LEGITIMATE_LABEL) & (preds == SCAM_LABEL)).sum()),
        "fp_url": int(
            ((labels == LEGITIMATE_LABEL) & (preds == SCAM_LABEL) & has_url).sum()
        ),
        "fp_no_url": int(
            ((labels == LEGITIMATE_LABEL) & (preds == SCAM_LABEL) & ~has_url).sum()
        ),
    }


# Collect FN and FP records plus slice summaries for one labeled split.
def analyze_link_errors(
    texts: pd.Series | list[str],
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float,
    split_name: str,
) -> dict[str, Any]:
    """Return FN/FP URL-flag records and slice metrics for one frozen split."""

    # Materialize texts for per-row feature extraction.
    text_list = as_text_list(texts)
    # Cast labels to integers matching the schema.
    labels = np.asarray(as_label_list(y_true), dtype=int)
    # Apply the VAL-frozen threshold; never search it here.
    preds = (np.asarray(y_proba) >= float(threshold)).astype(int)
    # Identify missed scams and warned ham.
    error_indices = [
        index
        for index, (gold, pred) in enumerate(zip(labels, preds, strict=True))
        if gold != pred
    ]
    # Build one audit row per error (FN or FP).
    records = [
        error_record(
            text_list[index],
            y_true=int(labels[index]),
            y_pred=int(preds[index]),
            p_scam=float(y_proba[index]),
            row_index=index,
        )
        for index in error_indices
    ]
    # Split records by error type for easier JSON inspection.
    false_negatives = [row for row in records if row["error_type"] == "FN"]
    # Warned ham rows (false positives).
    false_positives = [row for row in records if row["error_type"] == "FP"]
    # Count extra URL-related FNs (has_url or any lexical phishing flag).
    def _url_related(row: dict[str, Any]) -> bool:
        # A miss is URL-related when a link is present or a phishing flag fired.
        return bool(row["has_url"]) or int(row["suspicious_url_flag_count"]) > 0

    # Bundle everything the comparison/decision writers need.
    return {
        "split_name": split_name,
        "threshold": float(threshold),
        "slices": summarize_url_slices(text_list, labels, preds),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "fn_url_related": int(sum(_url_related(row) for row in false_negatives)),
        "fn_no_url_social": int(
            sum(not _url_related(row) for row in false_negatives)
        ),
        "fp_url_related": int(sum(_url_related(row) for row in false_positives)),
        "fp_link_heavy": int(sum(bool(row["link_heavy"]) for row in false_positives)),
    }


# Apply the documented A∧B∧C conjunction for a char-LSTM go/no-go.
def recommend_char_lstm_exploration(
    *,
    chat_eval_fn_lstm: int,
    chat_eval_fn_tfidf: int,
    test_fn_lstm: int,
    test_fn_distilbert: int,
    test_scam_recall_lstm: float,
    test_scam_recall_distilbert: float,
    test_scam_recall_tfidf: float,
    chat_eval_scam_recall_lstm: float,
    chat_eval_ham_warned_lstm: int,
    chat_eval_ham_warned_tfidf: int,
    extra_fn_chat_eval_url_related: int,
    extra_fn_test_url_related: int,
    lstm_url_scam_recall_chat: float | None,
    lstm_url_scam_recall_test: float | None,
    tfidf_url_scam_recall_chat: float | None,
    tfidf_url_scam_recall_test: float | None,
    tfidf_scam_recall_chat: float,
    chat_eval_fn_url: int,
    chat_eval_fn_no_url: int,
    test_fn_url: int,
    test_fn_no_url: int,
) -> dict[str, Any]:
    """Return a go/no-go payload; implement char LSTM only when A, B, and C all hold.

    A: chat-eval misses 10+ more scams than TF-IDF, OR TEST scam recall is worse
       than DistilBERT by CHAR_LSTM_A_TEST_RECALL_GAP or more.
    B: a majority of the extra missed scams vs TF-IDF (prefer chat-eval) are
       URL-bearing and/or carry shortener / suspicious TLD / IP / login-path flags.
    C: URL-bearing scam recall is still materially below TF-IDF's URL-bearing
       slice, or TF-IDF overall scam recall when that slice is unavailable.
    """

    # Extra chat-eval misses versus the published TF-IDF point (usually 0 FN).
    extra_chat = int(chat_eval_fn_lstm) - int(chat_eval_fn_tfidf)
    # Extra TEST misses versus DistilBERT (positive means LSTM missed more).
    extra_test_vs_distilbert = int(test_fn_lstm) - int(test_fn_distilbert)
    # Criterion A, chat-eval branch: 10+ extra missed scams vs TF-IDF.
    a_chat = extra_chat >= CHAR_LSTM_A_CHAT_EXTRA_FN
    # Criterion A, TEST branch: clearly large recall gap vs DistilBERT.
    a_test = (float(test_scam_recall_distilbert) - float(test_scam_recall_lstm)) >= (
        CHAR_LSTM_A_TEST_RECALL_GAP
    )
    # A holds if either documented failure mode is present.
    criterion_a = bool(a_chat or a_test)
    # Prefer chat-eval extra FNs for B when TF-IDF is the chat-eval ceiling.
    extra_chat_denom = max(extra_chat, 0)
    # Fraction of extra chat-eval FNs that are URL-related (0 if no extras).
    b_chat_frac = (
        float(extra_fn_chat_eval_url_related) / float(extra_chat_denom)
        if extra_chat_denom > 0
        else 0.0
    )
    # TEST extras vs TF-IDF FN count (published TEST TF-IDF FN is typically 27).
    extra_test_vs_tfidf = int(test_fn_lstm)  # filled by caller relative to TF-IDF below.
    # The caller passes extra_fn_test_url_related already counted on LSTM extra FNs.
    extra_test_denom = max(int(test_fn_lstm), 0)
    # TEST URL-related fraction among LSTM FNs (used when chat extras are 0).
    b_test_frac = (
        float(extra_fn_test_url_related) / float(extra_test_denom)
        if extra_test_denom > 0
        else 0.0
    )
    # Prefer chat-eval for B; fall back to TEST inspection when chat extras are 0.
    b_fraction = b_chat_frac if extra_chat_denom > 0 else b_test_frac
    # Criterion B: majority of those extra/inspected misses are URL-related.
    criterion_b = bool(b_fraction > 0.50)
    # TF-IDF URL-bearing reference: slice if present, else overall scam recall.
    tfidf_url_ref_chat = (
        float(tfidf_url_scam_recall_chat)
        if tfidf_url_scam_recall_chat is not None
        else float(tfidf_scam_recall_chat)
    )
    # Same reference on TEST.
    tfidf_url_ref_test = (
        float(tfidf_url_scam_recall_test)
        if tfidf_url_scam_recall_test is not None
        else float(test_scam_recall_tfidf)
    )
    # LSTM URL-bearing recall may be None when a split has no URL scams.
    lstm_url_chat = lstm_url_scam_recall_chat
    # TEST URL-bearing LSTM recall.
    lstm_url_test = lstm_url_scam_recall_test
    # Gap on chat-eval URL scams when both sides are defined.
    c_chat_gap = (
        tfidf_url_ref_chat - float(lstm_url_chat)
        if lstm_url_chat is not None
        else 0.0
    )
    # Gap on TEST URL scams when both sides are defined.
    c_test_gap = (
        tfidf_url_ref_test - float(lstm_url_test) if lstm_url_test is not None else 0.0
    )
    # Criterion C: URL concat did not close a material URL-scam recall gap.
    criterion_c = bool(
        c_chat_gap >= CHAR_LSTM_C_RECALL_GAP or c_test_gap >= CHAR_LSTM_C_RECALL_GAP
    )
    # Implement-now only when A, B, and C all hold.
    implement_now = bool(criterion_a and criterion_b and criterion_c)
    # Prefer chat-eval FN mix to decide "mostly social-engineering".
    chat_fn_total = int(chat_eval_fn_url) + int(chat_eval_fn_no_url)
    # Misses are mostly no-URL when no-URL FNs strictly outnumber URL FNs.
    mostly_no_url = bool(
        (chat_fn_total > 0 and int(chat_eval_fn_no_url) > int(chat_eval_fn_url))
        or (
            chat_fn_total == 0
            and int(test_fn_no_url) > int(test_fn_url)
            and int(test_fn_lstm) > 0
        )
    )
    # URL-bearing recall is "close" to TF-IDF when the gap is below C's material cut.
    url_recall_close = bool(
        (lstm_url_chat is None or c_chat_gap < CHAR_LSTM_C_RECALL_GAP)
        and (lstm_url_test is None or c_test_gap < CHAR_LSTM_C_RECALL_GAP)
    )
    # Reasonable third baseline: in-domain near DistilBERT/TF-IDF, quieter chat FPs.
    near_in_domain = bool(
        abs(float(test_scam_recall_lstm) - float(test_scam_recall_distilbert))
        <= CHAR_LSTM_NEAR_DISTILBERT_TEST
        or abs(float(test_scam_recall_lstm) - float(test_scam_recall_tfidf))
        <= CHAR_LSTM_NEAR_DISTILBERT_TEST
    )
    # Chat false alarms better than TF-IDF's 70/100 without collapsing scam recall.
    quieter_chat_fp = int(chat_eval_ham_warned_lstm) < int(chat_eval_ham_warned_tfidf)
    # Chat scam recall still usable (not a collapse toward 0).
    usable_chat_recall = float(chat_eval_scam_recall_lstm) >= CHAR_LSTM_USABLE_CHAT_SCAM_RECALL
    # Combined "already a reasonable third baseline" stop.
    reasonable_third = bool(near_in_domain and quieter_chat_fp and usable_chat_recall)
    # Only-pain-is-ham-FPs: few FNs and FPs dominate the remaining pain.
    only_ham_false_alarms = bool(
        int(chat_eval_fn_lstm) < CHAR_LSTM_A_CHAT_EXTRA_FN
        and int(chat_eval_ham_warned_lstm) > 0
        and not criterion_a
    )
    # Any documented no-go reason vetoes exploration even if A looked noisy.
    do_not_explore = bool(
        (not implement_now)
        or mostly_no_url
        or url_recall_close
        or reasonable_third
        or only_ham_false_alarms
    )
    # Final verdict: explore only when implement_now and no veto fired.
    explore = bool(implement_now and not (
        mostly_no_url or url_recall_close or reasonable_third or only_ham_false_alarms
    ))
    # Human-readable verdict string for the markdown file and agent summary.
    verdict = "explore_char_lstm" if explore else "do_not_explore_char_lstm"
    # Bundle every predicate so the markdown can cite A/B/C with numbers.
    return {
        "verdict": verdict,
        "implement_now": implement_now,
        "do_not_explore": do_not_explore,
        "criterion_a": criterion_a,
        "criterion_a_chat_extra_fn": extra_chat,
        "criterion_a_chat": a_chat,
        "criterion_a_test": a_test,
        "criterion_a_test_recall_gap": float(test_scam_recall_distilbert)
        - float(test_scam_recall_lstm),
        "criterion_a_extra_test_fn_vs_distilbert": extra_test_vs_distilbert,
        "criterion_b": criterion_b,
        "criterion_b_fraction": b_fraction,
        "criterion_b_used_split": "chat_eval" if extra_chat_denom > 0 else "test",
        "criterion_c": criterion_c,
        "criterion_c_chat_gap": c_chat_gap,
        "criterion_c_test_gap": c_test_gap,
        "mostly_no_url": mostly_no_url,
        "url_recall_close": url_recall_close,
        "reasonable_third_baseline": reasonable_third,
        "only_ham_false_alarms": only_ham_false_alarms,
        "extra_fn_chat_eval": extra_chat,
        "extra_fn_chat_eval_url_related": int(extra_fn_chat_eval_url_related),
        "extra_test_vs_tfidf_placeholder": extra_test_vs_tfidf,
        "notes": {
            "a_chat_extra_fn_threshold": CHAR_LSTM_A_CHAT_EXTRA_FN,
            "a_test_recall_gap_threshold": CHAR_LSTM_A_TEST_RECALL_GAP,
            "c_recall_gap_threshold": CHAR_LSTM_C_RECALL_GAP,
        },
    }


# Render the go/no-go markdown the user asked for (no char-LSTM code in this pass).
def render_char_lstm_decision_markdown(recommendation: dict[str, Any]) -> str:
    """Return markdown explaining explore vs do-not-explore from A/B/C."""

    # Pretty-print booleans as YES/NO for the decision file.
    def _yn(value: object) -> str:
        # True predicates render as YES so the conjunction is scannable.
        return "YES" if bool(value) else "NO"

    # Choose the heading from the verdict string.
    explore = recommendation["verdict"] == "explore_char_lstm"
    # Title states the recommendation up front.
    title = (
        "# Char LSTM decision: EXPLORE"
        if explore
        else "# Char LSTM decision: DO NOT explore"
    )
    # Optional one-paragraph design only when the rule says explore.
    design = ""
    # When explore is recommended, still do not implement in this pass.
    if explore:
        # Keep the proposed design to one paragraph as requested.
        design = (
            "\n\n## Proposed char-LSTM design (not implemented)\n\n"
            "If a follow-up pass is approved, train a **unidirectional** char LSTM "
            "on raw characters (suggested `max_chars=256`, vocab = printable ASCII "
            "plus an UNK), hidden size 64–128, 1 layer, no bidirectional wrapper "
            "unless a later cost check shows the extra backward pass is cheap on "
            "device. Concatenate the same TRAIN-scaled URL feature vector and reuse "
            "this word-BiLSTM protocol (TRAIN-only fit, VAL-only threshold, TEST "
            "once, chat-eval predict-only). Do not start that architecture until "
            "the user confirms.\n"
        )
    # Cite A/B/C with the recorded predicates on short lines for ruff E501.
    a_line = (
        f"- **A (not competitive):** {_yn(recommendation['criterion_a'])}"
    )
    # Spell out chat-eval extra FN vs TF-IDF on its own line.
    a_chat_line = (
        f"  Chat-eval extra FN vs TF-IDF = "
        f"{recommendation['criterion_a_chat_extra_fn']} "
        f"(need ≥ {recommendation['notes']['a_chat_extra_fn_threshold']}); "
        f"A_chat={_yn(recommendation['criterion_a_chat'])}."
    )
    # Spell out the TEST recall gap vs DistilBERT on its own line.
    a_test_line = (
        f"  TEST scam-recall gap vs DistilBERT = "
        f"{recommendation['criterion_a_test_recall_gap']:.4f} "
        f"(need ≥ {recommendation['notes']['a_test_recall_gap_threshold']}); "
        f"A_test={_yn(recommendation['criterion_a_test'])}."
    )
    # Criterion B header.
    b_line = (
        f"- **B (link-heavy extra misses):** {_yn(recommendation['criterion_b'])}"
    )
    # Criterion B fraction and extra-FN counts.
    b_detail = (
        f"  URL-related fraction on {recommendation['criterion_b_used_split']} = "
        f"{recommendation['criterion_b_fraction']:.3f} (need > 0.50). "
        f"Extra chat-eval FN = {recommendation['extra_fn_chat_eval']}; "
        f"URL-related = {recommendation['extra_fn_chat_eval_url_related']}."
    )
    # Criterion C header.
    c_line = (
        f"- **C (URL concat did not close the gap):** "
        f"{_yn(recommendation['criterion_c'])}"
    )
    # Criterion C gaps vs TF-IDF URL/overall recall.
    c_detail = (
        f"  Chat URL-recall gap vs TF-IDF ref = "
        f"{recommendation['criterion_c_chat_gap']:.4f}; "
        f"TEST URL-recall gap = {recommendation['criterion_c_test_gap']:.4f} "
        f"(material if ≥ {recommendation['notes']['c_recall_gap_threshold']})."
    )
    # Stop-condition bullets, each kept under 100 characters.
    stop_no_url = (
        f"- Misses mostly no-URL social-engineering: "
        f"{_yn(recommendation['mostly_no_url'])}"
    )
    # URL-recall-close stop.
    stop_url_close = (
        f"- URL-bearing scam recall already close to TF-IDF: "
        f"{_yn(recommendation['url_recall_close'])}"
    )
    # Reasonable-third-baseline stop.
    stop_third = (
        f"- Word BiLSTM already a reasonable third baseline: "
        f"{_yn(recommendation['reasonable_third_baseline'])}"
    )
    # Ham-false-alarm stop.
    stop_ham_fp = (
        f"- Remaining pain is ham false alarms on https links: "
        f"{_yn(recommendation['only_ham_false_alarms'])}"
    )
    # Assemble the markdown document from the short pieces.
    body = "\n".join(
        [
            title,
            "",
            "Word-level BiLSTM + TRAIN-scaled URL concat is the architecture",
            "in this pass. A character-level LSTM is **not** implemented here",
            "unless the documented conjunction A ∧ B ∧ C holds. It does **not**",
            "hold as an implement-now trigger unless every line below is YES.",
            "",
            "## Criteria",
            "",
            a_line,
            a_chat_line,
            a_test_line,
            b_line,
            b_detail,
            c_line,
            c_detail,
            "",
            f"Implement-now (A ∧ B ∧ C): **{_yn(recommendation['implement_now'])}**",
            "",
            "## Stop conditions (any one is enough to refuse char LSTM)",
            "",
            stop_no_url,
            stop_url_close,
            stop_third,
            stop_ham_fp,
            "",
            "## Verdict",
            "",
            f"**{recommendation['verdict']}**",
            "",
            "Do not export ONNX, do not wire the frontend, and do not train a",
            "char LSTM in this pass unless the user already asked and",
            "implement-now is YES.",
            design,
        ]
    )
    # Return the full markdown document.
    return body
