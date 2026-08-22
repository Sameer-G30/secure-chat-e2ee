# 05_epochs_5

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `epochs = 5`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.8998 / 0.9954 / 0.9452
- Legitimate precision / recall / F1: 0.9951 / 0.8949 / 0.9424
- Accuracy: 0.9438
- Combined mean (scam recall, ham precision, accuracy): 0.9781
- Scams missed / ham warned: 16 / 385
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3278, 385], [16, 3458]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `5`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9934
- Legitimate recall: 0.8976
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6522 / 0.9000 / 0.7563
- Legitimate precision / recall / F1: 0.8387 / 0.5200 / 0.6420
- Scams missed / ham warned: 10 / 48
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[52, 48], [10, 90]]`
