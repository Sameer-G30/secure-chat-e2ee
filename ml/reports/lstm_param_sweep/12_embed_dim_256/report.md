# 12_embed_dim_256

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `embed_dim = 256`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9407 / 0.9908 / 0.9651
- Legitimate precision / recall / F1: 0.9908 / 0.9408 / 0.9651
- Accuracy: 0.9651
- Combined mean (scam recall, ham precision, accuracy): 0.9822
- Scams missed / ham warned: 32 / 217
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3446, 217], [32, 3442]]`

## Training knobs

- `embed_dim`: `256`
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
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9889
- Legitimate recall: 0.9414
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7402 / 0.9400 / 0.8282
- Legitimate precision / recall / F1: 0.9178 / 0.6700 / 0.7746
- Scams missed / ham warned: 6 / 33
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[67, 33], [6, 94]]`
