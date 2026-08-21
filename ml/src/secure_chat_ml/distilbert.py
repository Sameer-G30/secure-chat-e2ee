"""Fine-tune DistilBERT for scam vs legitimate chat-register classification.

This module holds testable, importable logic; `scripts/train_distilbert.py`
is the thin CLI that points it at the real 71k LLM-rewritten rows.
Unit tests construct a tiny random DistilBERT from a local vocab file and
never download `distilbert-base-uncased`.

Training protocol (mirrors the TF-IDF baseline):
1. Fit on TRAIN only (never val/test/chat_eval).
2. Search only the documented decision-threshold grid on VALIDATION.
3. Selection rule: maximize scam recall subject to legitimate recall >= 0.85.
4. Freeze the threshold, score TEST once, then score the locked chat eval
   set predict-only (never fit or retune on those 200 rows).
5. Do not export ONNX here; browser ORT Web load is a later slice.
"""

# Import dataclass to bundle hyperparameters and threshold-tuning results.
from dataclasses import dataclass, field

# Import Path for typed checkpoint and report locations.
from pathlib import Path

# Import Any for sklearn classification_report dictionaries.
from typing import Any

# Import numpy for probability vectors and class-weight arrays.
import numpy as np

# Import pandas so callers can pass Series from the shared corpus loaders.
import pandas as pd

# Import PyTorch for tensors, devices, and the Dataset protocol.
import torch

# Import sklearn's balanced class-weight helper to match the TF-IDF head.
from sklearn.utils.class_weight import compute_class_weight

# Import Dataset so HuggingFace Trainer can iterate tokenized rows.
from torch.utils.data import Dataset

# Import HuggingFace classification pieces used by the live training script.
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

# Reuse the baseline split/eval/threshold rule so the two tracks stay comparable.
from secure_chat_ml.baseline import (
    CLASS_NAMES,
    DEFAULT_LEGIT_RECALL_FLOOR,
    DEFAULT_SELECTION_RULE,
    DEFAULT_THRESHOLD_GRID,
    LEGITIMATE_LABEL,
    SCAM_LABEL,
    BaselineEvaluation,
    evaluation_from_predictions,
    pick_operating_point,
    score_threshold_grid,
)

# HuggingFace Hub id for DistilBERT-base; tests never call from_pretrained on this.
DEFAULT_MODEL_NAME = "distilbert-base-uncased"

# Truncate at 256 WordPiece tokens: 600-char DMs plus URLs fit with little loss.
DEFAULT_MAX_LENGTH = 256

# Per-device train batch; fp16 on an 8 GB RTX 4060 holds DistilBERT at 256 tokens.
DEFAULT_TRAIN_BATCH_SIZE = 16

# Larger eval batches are safe because inference has no optimizer state.
DEFAULT_EVAL_BATCH_SIZE = 32

# Standard HuggingFace sequence-classification learning rate for DistilBERT.
DEFAULT_LEARNING_RATE = 2e-5

# Three epochs is the documented default; not searched on TEST or chat_eval.
DEFAULT_NUM_TRAIN_EPOCHS = 3

# Warm up 10% of steps so the first updates do not jump from a cold pretrained head.
DEFAULT_WARMUP_RATIO = 0.1

# Light AdamW decay, matching common DistilBERT classification recipes.
DEFAULT_WEIGHT_DECAY = 0.01

# VAL P(scam) grid for DistilBERT sweeps: documented 0.30..0.70 plus 0.20 and 0.25.
DISTILBERT_EXPANDED_THRESHOLD_GRID: tuple[float, ...] = tuple(
    i / 100 for i in range(20, 71, 5)
)

# Map integer labels onto the shared schema names stored in the HF config.
ID2LABEL = {LEGITIMATE_LABEL: CLASS_NAMES[0], SCAM_LABEL: CLASS_NAMES[1]}

# Inverse map so from_pretrained round-trips the same class order.
LABEL2ID = {CLASS_NAMES[0]: LEGITIMATE_LABEL, CLASS_NAMES[1]: SCAM_LABEL}


# Bundle the documented DistilBERT hyperparameters (frozen except threshold).
@dataclass(frozen=True)
class DistilBertHyperparameters:
    """Represent the DistilBERT knobs recorded in reports (not searched on TEST)."""

    # Record the Hub id or local directory the tokenizer/model were loaded from.
    model_name: str = DEFAULT_MODEL_NAME
    # Record the WordPiece truncation length applied at train and inference time.
    max_length: int = DEFAULT_MAX_LENGTH
    # Record the per-device training batch size actually used after any OOM retry.
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    # Record the per-device inference batch size.
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    # Record the AdamW learning rate.
    learning_rate: float = DEFAULT_LEARNING_RATE
    # Record how many epochs TRAIN was shown (VAL is not used as extra epochs).
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    # Record the linear-warmup fraction of optimizer steps.
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    # Record AdamW weight decay.
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    # Record the seed applied to torch/numpy/transformers.
    seed: int = 42


# Bundle VAL threshold selection so TEST and chat-eval reuse one frozen cut.
@dataclass
class DistilBertThresholdResult:
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


# Hold tokenized input_ids/attention_mask plus integer labels for Trainer.
class EncodedTextDataset(Dataset):
    """Return one DistilBERT encoding plus its binary label per index."""

    # Store the tokenizer output dict (lists of token ids, not padded tensors).
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        # Keep encodings aligned with labels by construction.
        self.encodings = encodings
        # Store labels as a plain list so __getitem__ can box a Python int.
        self.labels = labels

    # Report dataset length so DataLoader knows how many batches to build.
    def __len__(self) -> int:
        # Length follows the label vector, which matches encoding rows.
        return len(self.labels)

    # Return an unpadded example; DataCollatorWithPadding pads within the batch.
    def __getitem__(self, index: int) -> dict[str, list[int] | int]:
        # Copy input_ids for this row.
        item: dict[str, list[int] | int] = {
            "input_ids": self.encodings["input_ids"][index],
            # Copy the attention mask so padding tokens are ignored later.
            "attention_mask": self.encodings["attention_mask"][index],
            # Attach the integer class label expected by CrossEntropyLoss.
            "labels": int(self.labels[index]),
        }
        # Return the collator-ready example.
        return item


# Subclass Trainer so the loss uses TRAIN-only balanced class weights.
class WeightedTrainer(Trainer):
    """Trainer whose CrossEntropyLoss uses sklearn-balanced class weights."""

    # Accept class weights as a keyword so Trainer's (model, args, ...) stay intact.
    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        # Construct the upstream Trainer (model, args, datasets, collator, ...).
        super().__init__(*args, **kwargs)
        # Keep weights on CPU; compute_loss moves them next to the logits.
        self._class_weights = class_weights

    # Override the default mean cross-entropy so minority scams are not drowned.
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # Pop labels so the backbone forward does not compute its own unweighted loss.
        labels = inputs.pop("labels")
        # Run DistilBERT and the classification head to obtain logits.
        outputs = model(**inputs)
        # Read the [batch, 2] logit tensor from the sequence-classification output.
        logits = outputs.logits
        # Move TRAIN-fitted weights onto the same device/dtype as the logits.
        weight = self._class_weights.to(device=logits.device, dtype=logits.dtype)
        # Use mean-reduced weighted CE; ignore num_items_in_batch (sum/mean mismatch).
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        # Flatten in case a future collator adds an extra sequence dimension.
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        # Honor Trainer's (loss, outputs) contract when return_outputs is set.
        if return_outputs:
            # Pair the scalar loss with the model outputs for eval logging.
            return loss, outputs
        # Default training path returns the scalar loss only.
        return loss


# Convert a pandas Series or list into a plain list of strings.
def as_text_list(texts: pd.Series | list[str] | np.ndarray) -> list[str]:
    """Return message texts as `list[str]` for the HuggingFace tokenizer."""

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
    # numpy arrays need a Python list for EncodedTextDataset.
    if isinstance(labels, np.ndarray):
        # Flatten then int-cast each cell.
        return [int(label) for label in labels.tolist()]
    # A Python list is already usable.
    return [int(label) for label in labels]


# Seed numpy, Python, and torch so TRAIN fine-tunes are reproducible.
def set_reproducible_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) with the split seed."""

    # Import random locally so the library module does not need it at import time.
    import random

    # Seed Python's hash-adjacent RNG used by some collators.
    random.seed(seed)
    # Seed NumPy used by class-weight computation and metric helpers.
    np.random.seed(seed)
    # Seed PyTorch CPU generators.
    torch.manual_seed(seed)
    # Seed every CUDA device when a GPU is visible.
    if torch.cuda.is_available():
        # Keep GPU dropout and shuffling aligned with the CPU seed.
        torch.cuda.manual_seed_all(seed)


# Describe the device and whether fp16 is actually enabled.
def resolve_training_device() -> tuple[torch.device, bool, str]:
    """Return (device, use_fp16, reason) for this machine.

    fp16 is used only when CUDA is available. CPU training stays fp32 because
    HuggingFace Trainer's fp16 path requires CUDA GradScaler.
    """

    # Prefer the RTX 4060 when the CUDA driver is visible to this WSL2 process.
    if torch.cuda.is_available():
        # Build a cuda:0 device handle for model.to(...) and inference.
        device = torch.device("cuda")
        # fp16 fits DistilBERT-base at batch 16 / seq 256 on 8 GB.
        return device, True, "cuda_fp16"
    # Fall back to CPU fp32 when no GPU is present (unit tests, CI).
    device = torch.device("cpu")
    # Document that fp16 was skipped rather than silently training in fp32.
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
    # Return float32 CPU weights; WeightedTrainer moves them onto the logits device.
    return torch.tensor(weights, dtype=torch.float32)


# Tokenize a list of DMs with truncation but without padding.
def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Return unpadded input_ids and attention_mask lists for EncodedTextDataset."""

    # Truncate to max_length; padding happens later inside each Trainer batch.
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    # Keep only the tensors DistilBERT needs (drop token_type_ids if present).
    return {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded["attention_mask"]),
    }


# Count how many texts would exceed max_length without truncation.
def count_truncated_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
) -> int:
    """Return how many messages overflow `max_length` WordPiece tokens."""

    # Encode without truncation so we can see the true length.
    original_max = tokenizer.model_max_length
    # Raise the tokenizer cap so overflow counting does not warn at 512 tokens.
    tokenizer.model_max_length = 10**9
    try:
        # Tokenize every text at full length for the overflow count only.
        encoded = tokenizer(texts, truncation=False, padding=False)
    finally:
        # Restore DistilBERT's 512-token cap before training/inference.
        tokenizer.model_max_length = original_max
    # Count rows whose token length is strictly greater than the training cap.
    return sum(len(ids) > max_length for ids in encoded["input_ids"])


# Load pretrained DistilBERT plus its WordPiece tokenizer from Hub or disk.
def load_pretrained_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Return (model, tokenizer) for sequence classification with two labels.

    Live training calls this with `distilbert-base-uncased`. Tests must not:
    they construct a tiny random DistilBERT from a local vocab file instead.
    """

    # Load the WordPiece tokenizer matching the checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load DistilBERT with a 2-way classification head randomly initialized.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    # Ensure padding tokens are ignored inside the attention mask.
    if tokenizer.pad_token_id is not None:
        # DistilBERT uses [PAD]; copy its id onto the classification config.
        model.config.pad_token_id = tokenizer.pad_token_id
    # Return the pair the Trainer and inference helpers both need.
    return model, tokenizer


# Fine-tune an already-constructed model on TRAIN texts/labels only.
def fine_tune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_texts: pd.Series | list[str],
    train_labels: pd.Series | list[int],
    *,
    output_dir: Path,
    hyperparams: DistilBertHyperparameters | None = None,
    class_weights: torch.Tensor | None = None,
) -> PreTrainedModel:
    """Fit DistilBERT on TRAIN only and return the trained model (in place)."""

    # Fall back to documented defaults when the CLI did not override knobs.
    hyperparams = hyperparams or DistilBertHyperparameters()
    # Seed every RNG before shuffling or dropout runs.
    set_reproducible_seed(hyperparams.seed)
    # Resolve CUDA vs CPU and whether fp16 is actually enabled.
    device, use_fp16, _fp16_reason = resolve_training_device()
    # Materialize texts as a list for the tokenizer.
    text_list = as_text_list(train_texts)
    # Materialize labels as ints matching the schema.
    label_list = as_label_list(train_labels)
    # Compute balanced weights from TRAIN if the caller did not pass them.
    if class_weights is None:
        # Fit weights here so tests can also pass an explicit tensor.
        class_weights = balanced_class_weights(label_list)
    # Tokenize TRAIN once; DataCollatorWithPadding pads per batch.
    encodings = tokenize_texts(tokenizer, text_list, hyperparams.max_length)
    # Wrap encodings as a Dataset Trainer can iterate.
    train_dataset = EncodedTextDataset(encodings, label_list)
    # Pad to the longest sequence in each batch rather than to max_length.
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Ensure the output directory exists before Trainer writes logs.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Disable wandb/hub reporting so offline training does not prompt for tokens.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hyperparams.num_train_epochs,
        per_device_train_batch_size=hyperparams.train_batch_size,
        per_device_eval_batch_size=hyperparams.eval_batch_size,
        learning_rate=hyperparams.learning_rate,
        weight_decay=hyperparams.weight_decay,
        warmup_ratio=hyperparams.warmup_ratio,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        fp16=use_fp16,
        bf16=False,
        seed=hyperparams.seed,
        data_seed=hyperparams.seed,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=device.type == "cuda",
        use_cpu=device.type == "cpu",
        remove_unused_columns=False,
    )
    # Build the weighted Trainer; VAL is intentionally not passed as eval_dataset.
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        class_weights=class_weights,
    )
    # Run the TRAIN-only fine-tune.
    trainer.train()
    # Put the trained weights on the inference device before returning.
    model.to(device)
    # Switch to eval so later predict_scam_proba does not apply dropout.
    model.eval()
    # Return the same model object now holding fine-tuned weights.
    return model


# Score scam-class softmax probabilities without updating weights.
def predict_scam_proba(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: pd.Series | list[str],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    device: torch.device | None = None,
    use_fp16: bool | None = None,
) -> np.ndarray:
    """Return P(scam) for each text using a frozen DistilBERT classifier."""

    # Resolve device the same way training did when the caller omitted it.
    resolved_device, resolved_fp16, _reason = resolve_training_device()
    # Honor an explicit device (tests pin CPU even if CUDA exists).
    device = device if device is not None else resolved_device
    # Honor an explicit fp16 flag; default follows CUDA availability.
    use_fp16 = resolved_fp16 if use_fp16 is None else use_fp16
    # Materialize texts as a list so we can slice batches.
    text_list = as_text_list(texts)
    # Return an empty float array when there is nothing to score.
    if not text_list:
        # Keep dtype float64 to match numpy's default softmax promotion.
        return np.array([], dtype=np.float64)
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
            # Slice the current batch of raw strings.
            batch_texts = text_list[start : start + batch_size]
            # Pad this batch to its own longest sequence.
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            # Move token tensors onto the inference device.
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # Use fp16 autocast on CUDA; stay fp32 on CPU.
            if use_fp16 and device.type == "cuda":
                # Autocast DistilBERT matmuls to fp16 on the 4060.
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    # Forward pass yields [batch, 2] logits.
                    logits = model(**encoded).logits
            else:
                # CPU / fp32 path used by unit tests.
                logits = model(**encoded).logits
            # Convert logits to P(scam) = softmax[:, 1].
            batch_probs = torch.softmax(logits.float(), dim=-1)[:, SCAM_LABEL]
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
) -> DistilBertThresholdResult:
    """Return the VAL-frozen threshold; never looks at TEST or chat_eval."""

    # Cast labels to a 1-d numpy array for sklearn.
    labels = np.asarray(as_label_list(y_true), dtype=int)
    # Score every documented threshold on the already-computed VAL probabilities.
    candidates = score_threshold_grid(labels, y_proba, threshold_grid, C=0.0)
    # Apply the same floor-then-F1-fallback rule as the TF-IDF baseline.
    chosen, selection_reason, floor_feasible = pick_operating_point(
        candidates, legit_recall_floor=legit_recall_floor
    )
    # Bundle the frozen cut plus the VAL metrics at that operating point.
    return DistilBertThresholdResult(
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
    y_pred = (y_proba >= threshold).astype(int)
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


# Persist tokenizer + classification head for later predict-only scoring.
def save_classifier(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
) -> None:
    """Write HuggingFace `save_pretrained` files under output_dir (gitignored)."""

    # Ensure the checkpoint directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save model weights/config (safetensors via transformers default).
    model.save_pretrained(output_dir)
    # Save the tokenizer so inference does not need a Hub download.
    tokenizer.save_pretrained(output_dir)


# Reload a locally saved DistilBERT classifier without touching the Hub.
def load_saved_classifier(
    model_dir: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a checkpoint written by `save_classifier` from disk only."""

    # Fail loudly when the training script has not been run yet.
    if not model_dir.exists():
        # Point the operator at the named training script, not pytest.
        raise FileNotFoundError(
            f"No DistilBERT checkpoint at {model_dir}. "
            "Run scripts/train_distilbert.py first (not pytest)."
        )
    # Load from the local directory; tests pass a tiny saved random model here.
    return load_pretrained_classifier(str(model_dir))


# Re-export the default selection-rule string so scripts do not retype it.
SELECTION_RULE = DEFAULT_SELECTION_RULE
