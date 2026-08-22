# 15_num_layers_2

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `num_layers = 2`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9579 / 0.9816 / 0.9696
- Legitimate precision / recall / F1: 0.9821 / 0.9590 / 0.9704
- Accuracy: 0.9700
- Combined mean (scam recall, ham precision, accuracy): 0.9779
- Scams missed / ham warned: 64 / 150
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3513, 150], [64, 3410]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `2`
- `dropout`: `0.3`
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

- Scam recall: 0.9781
- Legitimate recall: 0.9596
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7788 / 0.8800 / 0.8263
- Legitimate precision / recall / F1: 0.8621 / 0.7500 / 0.8021
- Scams missed / ham warned: 12 / 25
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[75, 25], [12, 88]]`
