# 31_combo_epochs_8__learning_rate_5e-3

Protocol: post-OFAT multi-knob combo retrain.

**Changed parameter:** `combo = {'epochs': 8, 'learning_rate': 0.005}`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9743 / 0.9824 / 0.9784
- Legitimate precision / recall / F1: 0.9832 / 0.9754 / 0.9793
- Accuracy: 0.9788
- Combined mean (scam recall, ham precision, accuracy): 0.9815
- Scams missed / ham warned: 61 / 90
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3573, 90], [61, 3413]]`

## Training knobs

- `embed_dim`: `128`
- `hidden_size`: `128`
- `num_layers`: `1`
- `dropout`: `0.3`
- `max_tokens`: `128`
- `max_vocab_size`: `25000`
- `batch_size`: `128`
- `learning_rate`: `0.005`
- `epochs`: `8`
- `weight_decay`: `0.0`
- `grad_clip`: `1.0`
- `class_weight`: `balanced`
- `url_features`: `True`

## VALIDATION (threshold search; not reported as the final number)

- Scam recall: 0.9800
- Legitimate recall: 0.9726
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.8113 / 0.8600 / 0.8350
- Legitimate precision / recall / F1: 0.8511 / 0.8000 / 0.8247
- Scams missed / ham warned: 14 / 20
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[80, 20], [14, 86]]`
