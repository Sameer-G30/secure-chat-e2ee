# 16_num_layers_3

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `num_layers = 3`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9694 / 0.9750 / 0.9722
- Legitimate precision / recall / F1: 0.9761 / 0.9708 / 0.9734
- Accuracy: 0.9728
- Combined mean (scam recall, ham precision, accuracy): 0.9746
- Scams missed / ham warned: 87 / 107
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3556, 107], [87, 3387]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `3`
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

- Scam recall: 0.9748
- Legitimate recall: 0.9679
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.8019 / 0.8500 / 0.8252
- Legitimate precision / recall / F1: 0.8404 / 0.7900 / 0.8144
- Scams missed / ham warned: 15 / 21
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[79, 21], [15, 85]]`
