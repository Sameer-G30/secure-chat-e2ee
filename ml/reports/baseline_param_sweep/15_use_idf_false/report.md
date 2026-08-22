# 15_use_idf_false

Protocol: one-factor-at-a-time TF-IDF baseline retrain.

**Changed parameter:** `use_idf = False`

All other training knobs stay at the published defaults.
VAL searches the widened C grid and thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after C and the threshold are frozen.

## Frozen operating point (VALIDATION only)

- Chosen C: `2.0`
- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (7,137 in-domain rows; scored once)

- Scam precision / recall / F1: 0.8981 / 0.9922 / 0.9428
- Legitimate precision / recall / F1: 0.9918 / 0.8933 / 0.9400
- Accuracy: 0.9414
- Combined mean (scam recall, ham precision, accuracy): 0.9752
- Scams missed / ham warned: 27 / 391
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3272, 391], [27, 3447]]`

## VALIDATION (threshold/C search; not reported as the final number)

- Scam recall: 0.9909
- Legitimate recall: 0.8946
- C grid: `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]`
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6282 / 0.9800 / 0.7656
- Legitimate precision / recall / F1: 0.9545 / 0.4200 / 0.5833
- Scams missed / ham warned: 2 / 58
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[42, 58], [2, 98]]`
