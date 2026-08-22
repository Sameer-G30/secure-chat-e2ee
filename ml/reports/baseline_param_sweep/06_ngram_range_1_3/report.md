# 06_ngram_range_1_3

Protocol: one-factor-at-a-time TF-IDF baseline retrain.

**Changed parameter:** `ngram_range = (1, 3)`

All other training knobs stay at the published defaults.
VAL searches the widened C grid and thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after C and the threshold are frozen.

## Frozen operating point (VALIDATION only)

- Chosen C: `1.0`
- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (7,137 in-domain rows; scored once)

- Scam precision / recall / F1: 0.8777 / 0.9960 / 0.9331
- Legitimate precision / recall / F1: 0.9956 / 0.8684 / 0.9277
- Accuracy: 0.9305
- Combined mean (scam recall, ham precision, accuracy): 0.9740
- Scams missed / ham warned: 14 / 482
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3181, 482], [14, 3460]]`

## VALIDATION (threshold/C search; not reported as the final number)

- Scam recall: 0.9929
- Legitimate recall: 0.8702
- C grid: `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]`
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.5848 / 1.0000 / 0.7380
- Legitimate precision / recall / F1: 1.0000 / 0.2900 / 0.4496
- Scams missed / ham warned: 0 / 71
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[29, 71], [0, 100]]`
