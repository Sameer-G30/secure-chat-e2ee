"""Knowledge distillation of the DistilBERT scam teacher into a much smaller BERT-family student.

Design notes:
- Soft targets come from the teacher scoring the PLAIN current message. Windowed (multi-turn)
  student inputs reuse those targets, so the student learns to look through conversation context
  instead of copying the teacher's own multi-turn weakness.
- Truncation keeps the END of the text (truncation_side="left"), matching the browser tokenizer.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from secure_chat_ml.distilbert import (
    ID2LABEL,
    LABEL2ID,
    balanced_class_weights,
    resolve_training_device,
    set_reproducible_seed,
)


def load_student(name_or_dir: str) -> tuple[Any, Any]:
    """Load a 2-label BERT-family student with eager attention and left truncation."""

    tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
    # Keep the tail so the current (last) message survives truncation, like the browser.
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_dir,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # Eager attention exports as plain MatMul/Softmax that ORT Web can run.
        attn_implementation="eager",
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


@torch.no_grad()
def teacher_logits(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Return [N, 2] fp32 teacher logits for `texts` (length-sorted batches for speed)."""

    device, use_fp16, _reason = resolve_training_device()
    model.to(device).eval()
    order = np.argsort([len(text) for text in texts])
    out = np.zeros((len(texts), 2), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        index = order[start : start + batch_size]
        encoded = tokenizer(
            [texts[i] for i in index],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if use_fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**encoded).logits
        else:
            logits = model(**encoded).logits
        out[index] = logits.float().cpu().numpy()
    return out


class DistillDataset(Dataset):
    """Student inputs with hard labels and the teacher's logits for the same message."""

    def __init__(
        self,
        encodings: dict[str, list[list[int]]],
        labels: list[int],
        soft_logits: np.ndarray,
    ) -> None:
        self.encodings = encodings
        self.labels = labels
        self.soft_logits = soft_logits

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": int(self.labels[index]),
            "teacher_logits": self.soft_logits[index].tolist(),
        }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits_batch: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """alpha * weighted CE(hard) + (1 - alpha) * T^2 * KL(teacher || student)."""

    hard = F.cross_entropy(student_logits, labels, weight=class_weights)
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits_batch / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    return alpha * hard + (1.0 - alpha) * soft


class DistillTrainer(Trainer):
    """Trainer that applies distillation_loss using the batch's teacher logits."""

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        alpha: float,
        temperature: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._alpha = alpha
        self._temperature = temperature

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        teacher = inputs.pop("teacher_logits").to(dtype=torch.float32)
        outputs = model(**inputs)
        logits = outputs.logits.float()
        weight = self._class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = distillation_loss(
            logits,
            teacher.to(logits.device),
            labels,
            weight,
            alpha=self._alpha,
            temperature=self._temperature,
        )
        return (loss, outputs) if return_outputs else loss


class TeacherAwareCollator(DataCollatorWithPadding):
    """Pad token fields normally and stack the teacher logits."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        teacher = torch.tensor([row.pop("teacher_logits") for row in features], dtype=torch.float32)
        batch = super().__call__(features)
        batch["teacher_logits"] = teacher
        return batch


def distill_student(
    student: Any,
    tokenizer: Any,
    texts: list[str],
    labels: list[int],
    soft_logits: np.ndarray,
    *,
    output_dir: Path,
    max_length: int = 128,
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 8e-5,
    alpha: float = 0.5,
    temperature: float = 2.0,
    seed: int = 42,
) -> Any:
    """Train `student` on (texts, labels, teacher logits) and return it in eval mode."""

    set_reproducible_seed(seed)
    device, use_fp16, _reason = resolve_training_device()
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    dataset = DistillDataset(
        {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]},
        labels,
        soft_logits,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=200,
        fp16=use_fp16,
        seed=seed,
        data_seed=seed,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=device.type == "cpu",
        remove_unused_columns=False,
    )
    trainer = DistillTrainer(
        model=student,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=TeacherAwareCollator(tokenizer=tokenizer),
        class_weights=balanced_class_weights(labels),
        alpha=alpha,
        temperature=temperature,
    )
    trainer.train()
    student.eval()
    return student
