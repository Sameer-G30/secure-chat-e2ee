# 18_dropout_0.2

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `dropout = 0.2`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9472 / 0.9856 / 0.9660
- Legitimate precision / recall / F1: 0.9858 / 0.9479 / 0.9665
- Accuracy: 0.9662
- Combined mean (scam recall, ham precision, accuracy): 0.9792
- Scams missed / ham warned: 50 / 191
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3472, 191], [50, 3424]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.2`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9852
- Legitimate recall: 0.9491
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7778 / 0.9100 / 0.8387
- Legitimate precision / recall / F1: 0.8916 / 0.7400 / 0.8087
- Scams missed / ham warned: 9 / 26
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[74, 26], [9, 91]]`
