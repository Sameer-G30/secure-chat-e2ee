# 06_epochs_6

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `epochs = 6`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9466 / 0.9893 / 0.9675
- Legitimate precision / recall / F1: 0.9894 / 0.9470 / 0.9678
- Accuracy: 0.9676
- Combined mean (scam recall, ham precision, accuracy): 0.9821
- Scams missed / ham warned: 37 / 194
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3469, 194], [37, 3437]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.001`
- `epochs`: `6`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9895
- Legitimate recall: 0.9398
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7087 / 0.9000 / 0.7930
- Legitimate precision / recall / F1: 0.8630 / 0.6300 / 0.7283
- Scams missed / ham warned: 10 / 37
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[63, 37], [10, 90]]`
