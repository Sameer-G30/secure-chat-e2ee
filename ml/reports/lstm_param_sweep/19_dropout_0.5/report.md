# 19_dropout_0.5

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `dropout = 0.5`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.8960 / 0.9899 / 0.9406
- Legitimate precision / recall / F1: 0.9894 / 0.8911 / 0.9377
- Accuracy: 0.9392
- Combined mean (scam recall, ham precision, accuracy): 0.9728
- Scams missed / ham warned: 35 / 399
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3264, 399], [35, 3439]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.5`
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

- Scam recall: 0.9911
- Legitimate recall: 0.8982
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6786 / 0.9500 / 0.7917
- Legitimate precision / recall / F1: 0.9167 / 0.5500 / 0.6875
- Scams missed / ham warned: 5 / 45
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[55, 45], [5, 95]]`
