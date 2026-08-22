# 29_class_weight_none

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `class_weight = none`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9448 / 0.9845 / 0.9642
- Legitimate precision / recall / F1: 0.9846 / 0.9454 / 0.9646
- Accuracy: 0.9644
- Combined mean (scam recall, ham precision, accuracy): 0.9778
- Scams missed / ham warned: 54 / 200
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3463, 200], [54, 3420]]`

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
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `none`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9843
- Legitimate recall: 0.9395
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7373 / 0.8700 / 0.7982
- Legitimate precision / recall / F1: 0.8415 / 0.6900 / 0.7582
- Scams missed / ham warned: 13 / 31
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[69, 31], [13, 87]]`
