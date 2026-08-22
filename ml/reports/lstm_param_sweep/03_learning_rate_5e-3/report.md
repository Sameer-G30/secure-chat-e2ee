# 03_learning_rate_5e-3

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `learning_rate = 0.005`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9514 / 0.9911 / 0.9708
- Legitimate precision / recall / F1: 0.9912 / 0.9520 / 0.9712
- Accuracy: 0.9710
- Combined mean (scam recall, ham precision, accuracy): 0.9844
- Scams missed / ham warned: 31 / 176
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3487, 176], [31, 3443]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.005`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9899
- Legitimate recall: 0.9537
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7586 / 0.8800 / 0.8148
- Legitimate precision / recall / F1: 0.8571 / 0.7200 / 0.7826
- Scams missed / ham warned: 12 / 28
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[72, 28], [12, 88]]`
