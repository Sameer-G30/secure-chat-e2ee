# 01_learning_rate_5e-4

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `learning_rate = 0.0005`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9115 / 0.9902 / 0.9492
- Legitimate precision / recall / F1: 0.9899 / 0.9088 / 0.9476
- Accuracy: 0.9484
- Combined mean (scam recall, ham precision, accuracy): 0.9762
- Scams missed / ham warned: 34 / 334
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3329, 334], [34, 3440]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.0005`
- `epochs`: `4`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9883
- Legitimate recall: 0.9111
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6593 / 0.8900 / 0.7574
- Legitimate precision / recall / F1: 0.8308 / 0.5400 / 0.6545
- Scams missed / ham warned: 11 / 46
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[54, 46], [11, 89]]`
