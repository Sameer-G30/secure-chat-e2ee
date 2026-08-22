# 17_dropout_0

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `dropout = 0.0`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9654 / 0.9796 / 0.9724
- Legitimate precision / recall / F1: 0.9803 / 0.9667 / 0.9735
- Accuracy: 0.9730
- Combined mean (scam recall, ham precision, accuracy): 0.9776
- Scams missed / ham warned: 71 / 122
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3541, 122], [71, 3403]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.0`
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

- Scam recall: 0.9774
- Legitimate recall: 0.9655
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.7925 / 0.8400 / 0.8155
- Legitimate precision / recall / F1: 0.8298 / 0.7800 / 0.8041
- Scams missed / ham warned: 16 / 22
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[78, 22], [16, 84]]`
