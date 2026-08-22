# 08_max_tokens_64

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `max_tokens = 64`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9466 / 0.9845 / 0.9651
- Legitimate precision / recall / F1: 0.9847 / 0.9473 / 0.9656
- Accuracy: 0.9654
- Combined mean (scam recall, ham precision, accuracy): 0.9782
- Scams missed / ham warned: 54 / 193
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3470, 193], [54, 3420]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `64`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9856
- Legitimate recall: 0.9462
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6953 / 0.8900 / 0.7807
- Legitimate precision / recall / F1: 0.8472 / 0.6100 / 0.7093
- Scams missed / ham warned: 11 / 39
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[61, 39], [11, 89]]`
