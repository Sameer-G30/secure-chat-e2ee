# 07_ngram_range_2_2

Protocol: one-factor-at-a-time TF-IDF baseline retrain.

**Changed parameter:** `ngram_range = (2, 2)`

All other training knobs stay at the published defaults.
VAL searches the widened C grid and thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after C and the threshold are frozen.

## Frozen operating point (VALIDATION only)

- Chosen C: `4.0`
- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (7,137 in-domain rows; scored once)

- Scam precision / recall / F1: 0.8885 / 0.9911 / 0.9370
- Legitimate precision / recall / F1: 0.9905 / 0.8821 / 0.9331
- Accuracy: 0.9351
- Combined mean (scam recall, ham precision, accuracy): 0.9722
- Scams missed / ham warned: 31 / 432
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3231, 432], [31, 3443]]`

## VALIDATION (threshold/C search; not reported as the final number)

- Scam recall: 0.9896
- Legitimate recall: 0.8821
- C grid: `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]`
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.5765 / 0.9800 / 0.7259
- Legitimate precision / recall / F1: 0.9333 / 0.2800 / 0.4308
- Scams missed / ham warned: 2 / 72
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[28, 72], [2, 98]]`
