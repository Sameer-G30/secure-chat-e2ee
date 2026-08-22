# 00_baseline_expanded_grid

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `threshold_grid = expanded_0.20_to_0.70_step_0.05`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9596 / 0.9842 / 0.9717
- Legitimate precision / recall / F1: 0.9846 / 0.9607 / 0.9725
- Accuracy: 0.9721
- Combined mean (scam recall, ham precision, accuracy): 0.9803
- Scams missed / ham warned: 55 / 144
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3519, 144], [55, 3419]]`

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
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9799
- Legitimate recall: 0.9590
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7458 / 0.8800 / 0.8073
- Legitimate precision / recall / F1: 0.8537 / 0.7000 / 0.7692
- Scams missed / ham warned: 12 / 30
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[70, 30], [12, 88]]`
