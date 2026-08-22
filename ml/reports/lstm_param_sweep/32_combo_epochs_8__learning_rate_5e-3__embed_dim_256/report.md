# 32_combo_epochs_8__learning_rate_5e-3__embed_dim_256

Protocol: post-OFAT multi-knob combo retrain.

**Changed parameter:** `combo = {'epochs': 8, 'learning_rate': 0.005, 'embed_dim': 256}`

VAL searches thresholds 0.20, 0.25, ..., 0.70.
TEST and the locked chat-eval set are scored after the threshold is frozen.
This folder does not overwrite `reports/lstm/`.

## Frozen operating point (VALIDATION only)

- Chosen threshold: `0.2`
- Floor feasible: `True`
- Selection reason: `max_scam_recall_subject_to_legit_recall_floor`

## TEST (in-domain rows; scored once)

- Scam precision / recall / F1: 0.9751 / 0.9807 / 0.9779
- Legitimate precision / recall / F1: 0.9816 / 0.9762 / 0.9789
- Accuracy: 0.9784
- Combined mean (scam recall, ham precision, accuracy): 0.9802
- Scams missed / ham warned: 67 / 87
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[3576, 87], [67, 3407]]`

## Training knobs

- `embed_dim`: `256`
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

- Scam recall: 0.9760
- Legitimate recall: 0.9739
- Threshold grid: `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`

## Locked chat-style eval (200 rows; predict-only; not used for ranking)

- Scam precision / recall / F1: 0.8269 / 0.8600 / 0.8431
- Legitimate precision / recall / F1: 0.8542 / 0.8200 / 0.8367
- Scams missed / ham warned: 14 / 18
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[82, 18], [14, 86]]`
