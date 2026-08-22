# 19_solver_saga

Protocol: one-factor-at-a-time TF-IDF baseline retrain.

**Changed parameter:** `solver = saga`

All other training knobs stay at the published defaults.
VAL searches the widened C grid and thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after C and the threshold are frozen.

## Frozen operating point (VALIDATION only)

- Chosen C: `0.1`
- Chosen threshold: `0.4`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (7,137 in-domain rows; scored once)

- Scam precision / recall / F1: 0.8677 / 0.9819 / 0.9213
- Legitimate precision / recall / F1: 0.9803 / 0.8580 / 0.9151
- Accuracy: 0.9183
- Combined mean (scam recall, ham precision, accuracy): 0.9602
- Scams missed / ham warned: 63 / 520
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3143, 520], [63, 3411]]`

## VALIDATION (threshold/C search; not reported as the final number)

- Scam recall: 0.9776
- Legitimate recall: 0.8621
- C grid: `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]`
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.5917 / 1.0000 / 0.7435
- Legitimate precision / recall / F1: 1.0000 / 0.3100 / 0.4733
- Scams missed / ham warned: 0 / 69
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[31, 69], [0, 100]]`
