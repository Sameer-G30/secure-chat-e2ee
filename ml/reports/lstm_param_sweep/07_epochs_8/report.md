# 07_epochs_8

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `epochs = 8`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9506 / 0.9914 / 0.9706
- Legitimate precision / recall / F1: 0.9915 / 0.9511 / 0.9709
- Accuracy: 0.9707
- Combined mean (scam recall, ham precision, accuracy): 0.9845
- Scams missed / ham warned: 30 / 179
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3484, 179], [30, 3444]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `8`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9875
- Legitimate recall: 0.9481
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7542 / 0.8900 / 0.8165
- Legitimate precision / recall / F1: 0.8659 / 0.7100 / 0.7802
- Scams missed / ham warned: 11 / 29
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[71, 29], [11, 89]]`
