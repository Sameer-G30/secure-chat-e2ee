# 21_max_vocab_size_15000

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `max_vocab_size = 15000`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9574 / 0.9824 / 0.9697
- Legitimate precision / recall / F1: 0.9829 / 0.9585 / 0.9706
- Accuracy: 0.9702
- Combined mean (scam recall, ham precision, accuracy): 0.9785
- Scams missed / ham warned: 61 / 152
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3511, 152], [61, 3413]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `15000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9827
- Legitimate recall: 0.9556
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7328 / 0.8500 / 0.7870
- Legitimate precision / recall / F1: 0.8214 / 0.6900 / 0.7500
- Scams missed / ham warned: 15 / 31
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[69, 31], [15, 85]]`
