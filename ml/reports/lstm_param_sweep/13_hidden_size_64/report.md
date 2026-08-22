# 13_hidden_size_64

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `hidden_size = 64`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9476 / 0.9787 / 0.9629
- Legitimate precision / recall / F1: 0.9791 / 0.9487 / 0.9637
- Accuracy: 0.9633
- Combined mean (scam recall, ham precision, accuracy): 0.9737
- Scams missed / ham warned: 74 / 188
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3475, 188], [74, 3400]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `64`
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

- Scam recall: 0.9793
- Legitimate recall: 0.9483
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6911 / 0.8500 / 0.7623
- Legitimate precision / recall / F1: 0.8052 / 0.6200 / 0.7006
- Scams missed / ham warned: 15 / 38
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[62, 38], [15, 85]]`
