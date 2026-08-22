# 20_max_vocab_size_10000

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `max_vocab_size = 10000`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9349 / 0.9885 / 0.9610
- Legitimate precision / recall / F1: 0.9885 / 0.9348 / 0.9609
- Accuracy: 0.9609
- Combined mean (scam recall, ham precision, accuracy): 0.9793
- Scams missed / ham warned: 40 / 239
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3424, 239], [40, 3434]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `10000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9883
- Legitimate recall: 0.9360
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7165 / 0.9100 / 0.8018
- Legitimate precision / recall / F1: 0.8767 / 0.6400 / 0.7399
- Scams missed / ham warned: 9 / 36
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[64, 36], [9, 91]]`
