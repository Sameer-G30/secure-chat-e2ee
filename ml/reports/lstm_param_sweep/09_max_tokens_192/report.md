# 09_max_tokens_192

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `max_tokens = 192`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9378 / 0.9888 / 0.9626
- Legitimate precision / recall / F1: 0.9888 / 0.9378 / 0.9626
- Accuracy: 0.9626
- Combined mean (scam recall, ham precision, accuracy): 0.9800
- Scams missed / ham warned: 39 / 228
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3435, 228], [39, 3435]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `192`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9873
- Legitimate recall: 0.9357
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7385 / 0.9600 / 0.8348
- Legitimate precision / recall / F1: 0.9429 / 0.6600 / 0.7765
- Scams missed / ham warned: 4 / 34
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[66, 34], [4, 96]]`
