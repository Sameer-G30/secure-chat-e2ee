# 02_learning_rate_2e-3

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `learning_rate = 0.002`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9409 / 0.9896 / 0.9646
- Legitimate precision / recall / F1: 0.9897 / 0.9410 / 0.9647
- Accuracy: 0.9647
- Combined mean (scam recall, ham precision, accuracy): 0.9813
- Scams missed / ham warned: 36 / 216
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3447, 216], [36, 3438]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.002`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9882
- Legitimate recall: 0.9432
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7438 / 0.9000 / 0.8145
- Legitimate precision / recall / F1: 0.8734 / 0.6900 / 0.7709
- Scams missed / ham warned: 10 / 31
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[69, 31], [10, 90]]`
