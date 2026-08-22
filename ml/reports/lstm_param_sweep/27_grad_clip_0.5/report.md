# 27_grad_clip_0.5

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `grad_clip = 0.5`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9594 / 0.9856 / 0.9723
- Legitimate precision / recall / F1: 0.9860 / 0.9604 / 0.9730
- Accuracy: 0.9727
- Combined mean (scam recall, ham precision, accuracy): 0.9814
- Scams missed / ham warned: 50 / 145
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3518, 145], [50, 3424]]`

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
- `grad_clip`: `0.5`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9823
- Legitimate recall: 0.9559
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7120 / 0.8900 / 0.7911
- Legitimate precision / recall / F1: 0.8533 / 0.6400 / 0.7314
- Scams missed / ham warned: 11 / 36
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[64, 36], [11, 89]]`
