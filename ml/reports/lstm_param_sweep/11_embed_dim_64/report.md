# 11_embed_dim_64

Protocol: one-factor-at-a-time word-BiLSTM retrain.

**Changed parameter:** `embed_dim = 64`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9486 / 0.9830 / 0.9655
- Legitimate precision / recall / F1: 0.9833 / 0.9495 / 0.9661
- Accuracy: 0.9658
- Combined mean (scam recall, ham precision, accuracy): 0.9774
- Scams missed / ham warned: 59 / 185
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3478, 185], [59, 3415]]`

## Training knobs

- `embed_dim`: `64`
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

- Scam recall: 0.9824
- Legitimate recall: 0.9457
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.6855 / 0.8500 / 0.7589
- Legitimate precision / recall / F1: 0.8026 / 0.6100 / 0.6932
- Scams missed / ham warned: 15 / 39
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[61, 39], [15, 85]]`
