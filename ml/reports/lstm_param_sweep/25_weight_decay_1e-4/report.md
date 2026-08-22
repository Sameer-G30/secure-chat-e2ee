# 25_weight_decay_1e-4

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `weight_decay = 0.0001`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9520 / 0.9712 / 0.9615
- Legitimate precision / recall / F1: 0.9722 / 0.9536 / 0.9628
- Accuracy: 0.9622
- Combined mean (scam recall, ham precision, accuracy): 0.9685
- Scams missed / ham warned: 100 / 170
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3493, 170], [100, 3374]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `4`
- `weight_decay`: `0.0001`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9702
- Legitimate recall: 0.9528
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6569 / 0.9000 / 0.7595
- Legitimate precision / recall / F1: 0.8413 / 0.5300 / 0.6503
- Scams missed / ham warned: 10 / 47
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[53, 47], [10, 90]]`
