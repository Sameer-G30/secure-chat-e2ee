# Char LSTM decision: DO NOT explore

Word-level BiLSTM + TRAIN-scaled URL concat is the architecture
in this pass. A character-level LSTM is **not** implemented here
unless the documented conjunction A ∧ B ∧ C holds. It does **not**
hold as an implement-now trigger unless every line below is YES.

## Criteria

- **A (not competitive):** YES
  Chat-eval extra FN vs TF-IDF = 11 (need ≥ 10); A_chat=YES.
  TEST scam-recall gap vs DistilBERT = -0.0029 (need ≥ 0.05); A_test=NO.
- **B (link-heavy extra misses):** NO
  URL-related fraction on chat_eval = 0.000 (need > 0.50). Extra chat-eval FN = 11; URL-related = 0.
- **C (URL concat did not close the gap):** NO
  Chat URL-recall gap vs TF-IDF ref = 0.0000; TEST URL-recall gap = -0.0052 (material if ≥ 0.05).

Implement-now (A ∧ B ∧ C): **NO**

## Stop conditions (any one is enough to refuse char LSTM)

- Misses mostly no-URL social-engineering: YES
- URL-bearing scam recall already close to TF-IDF: YES
- Word BiLSTM already a reasonable third baseline: YES
- Remaining pain is ham false alarms on https links: NO

## Verdict

**do_not_explore_char_lstm**

Do not export ONNX, do not wire the frontend, and do not train a
char LSTM in this pass unless the user already asked and
implement-now is YES.
