# 30_url_features_false

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `url_features = False`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9519 / 0.9856 / 0.9685
- Legitimate precision / recall / F1: 0.9859 / 0.9528 / 0.9690
- Accuracy: 0.9688
- Combined mean (scam recall, ham precision, accuracy): 0.9801
- Scams missed / ham warned: 50 / 173
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3490, 173], [50, 3424]]`

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
- `url_features`: `False`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9830
- Legitimate recall: 0.9513
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7273 / 0.8800 / 0.7964
- Legitimate precision / recall / F1: 0.8481 / 0.6700 / 0.7486
- Scams missed / ham warned: 12 / 33
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[67, 33], [12, 88]]`
