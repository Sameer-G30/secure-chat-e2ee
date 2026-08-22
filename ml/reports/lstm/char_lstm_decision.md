# Char LSTM decision: DO NOT explore

Word-level BiLSTM + TRAIN-scaled URL concat is the architecture
in this pass. A character-level LSTM is **not** implemented here
unless the documented conjunction A ∧ B ∧ C holds. It does **not**.

## What the word BiLSTM actually did

Same 71,370-row `llm_intent_v1` corpus, same 70/20/10 split
(`random_state=42`), same VAL rule. TRAIN-only fit (vocab, embeddings,
BiLSTM, head, URL `StandardScaler`). VAL-frozen threshold **0.30**
(`max_scam_recall_subject_to_legit_recall_floor`, floor feasible).
fp32 on CUDA (`cuda_fp32_lstm_amp_unstable`); 39.5 s TRAIN wall-clock
on the RTX 4060. 930 TRAIN rows truncated at `max_tokens=128`.

| Split | Scam recall | Ham warned | Scams missed |
| --- | --- | --- | --- |
| TEST (7,137) | 0.986 | 200 / 3,663 | 49 / 3,474 |
| Chat eval (200) | 0.860 | 24 / 100 | 14 / 100 |

TF-IDF TEST misses 27 / chat-eval 0. DistilBERT TEST misses 60 /
chat-eval 12.

## Link-heavy FN/FP slices

On-device URL flags only (`live_url_reputation=false`). TF-IDF and
DistilBERT per-row errors were **not** recomputed (those checkpoints
were not loaded). Extra-FN counts subtract published FN totals.

**TEST:** overall scam recall 0.986; URL-bearing scam recall **0.995**
(4 FN / 786 URL scams); no-URL scam recall 0.983 (45 FN / 2,688).
Of 49 TEST FNs, **45 are no-URL** social/rewrite noise and 4 are
URL-bearing. Of 200 TEST FPs, 38 have a URL (26 link-heavy).

**Chat eval:** overall scam recall 0.860; URL-bearing scam recall
**1.000** (0 FN / 6 URL scams); no-URL scam recall 0.851 (14 FN / 94).
All **14 extra misses vs TF-IDF are no-URL** (“hi mom” / grandma bail /
crypto-without-a-link / KYC / seed-adjacent payroll). Chat FPs: 24,
of which 1 is URL-related and 1 is link-heavy — not a shortener/TLD
failure mode.

## Criteria A ∧ B ∧ C

- **A (not competitive):** YES on chat-eval only. Extra FN vs TF-IDF =
  14 (need ≥ 10); A_chat=YES. TEST scam-recall gap vs DistilBERT =
  **−0.003** (LSTM *better*, 49 vs 60 misses); A_test=NO (need ≥ 0.05).
- **B (link-heavy extra misses):** NO. URL-related fraction of extra
  chat-eval FNs = **0.000** (need > 0.50). Extra chat-eval FN = 14;
  URL-related = 0.
- **C (URL concat did not close the gap):** NO. Chat URL-recall gap vs
  TF-IDF overall (1.000) = **0.000** (LSTM URL scams 6/6). TEST
  URL-recall 0.995 vs TF-IDF overall 0.992 (gap −0.003). Material if
  ≥ 0.05.

Implement-now (A ∧ B ∧ C): **NO**

## Stop conditions (any one is enough to refuse char LSTM)

- Misses mostly no-URL social-engineering: **YES** (14/14 chat FNs;
  45/49 TEST FNs).
- URL-bearing scam recall already close to TF-IDF after concat: **YES**
  (chat URL scams 1.000; TEST URL scams 0.995).
- Word BiLSTM already a reasonable third baseline: **YES** (TEST near
  DistilBERT/TF-IDF; chat ham warned 24/100 vs TF-IDF 70/100; chat
  scam recall 0.860 is usable).
- Remaining pain is ham false alarms on ordinary https links: **NO**
  (chat FPs are mostly link-free chat; DistilBERT is quieter on ham).

## Verdict

**do_not_explore_char_lstm**

A character-level LSTM would model URL strings that this word model
already catches after URL-feature concat. The remaining misses are
no-URL social-engineering DMs; DistilBERT is the semantic model for
that gap. Do not export ONNX, do not wire the frontend, and do not
train a char LSTM unless a later pass revisits this rule with new
numbers.
