"""Score ONNX classifiers and compute the metrics used by the compression acceptance gate."""

# Import math for the McNemar chi-square tail probability.
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import average_precision_score, roc_auc_score


def score_onnx(
    onnx_path: Path,
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int = 32,
    threads: int = 4,
) -> np.ndarray:
    """Return P(scam) for each text using an ONNX sequence classifier on CPU."""

    # Limit threads so timings are stable and do not starve the machine.
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    # Some exports (e.g. DistilBERT) have no token_type_ids input; only feed what exists.
    input_names = {item.name for item in session.get_inputs()}
    # Sort by length so each batch pads as little as possible, then restore the order.
    order = np.argsort([len(text) for text in texts])
    probabilities = np.zeros(len(texts), dtype=np.float64)
    for start in range(0, len(texts), batch_size):
        index = order[start : start + batch_size]
        encoded = tokenizer(
            [texts[i] for i in index],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="np",
        )
        feeds = {name: encoded[name].astype(np.int64) for name in input_names if name in encoded}
        logits = session.run(None, feeds)[0].astype(np.float64)
        # Stable softmax; class 1 is "scam".
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities[index] = (exp / exp.sum(axis=1, keepdims=True))[:, 1]
    return probabilities


def metrics_at_threshold(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Return scam precision/recall/F1, accuracy, ROC-AUC, and PR-AUC at a frozen threshold."""

    y_pred = (proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # AUCs need both classes present.
    both = len(set(y_true.tolist())) == 2
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if both else float("nan"),
    }


def mcnemar_p_value(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Return the two-sided McNemar p-value (continuity corrected) for two classifiers.

    A small p means the classifiers disagree on correctness more than chance allows.
    """

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    # b = A right and B wrong; c = A wrong and B right.
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return 1.0
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    # Chi-square with 1 dof: survival function = erfc(sqrt(x/2)).
    return float(math.erfc(math.sqrt(statistic / 2)))
