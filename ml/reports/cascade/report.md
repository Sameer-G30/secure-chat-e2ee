# Cascade and ensemble report

chat_style_eval_v1.csv was never used for training, threshold search, 
cascade-edge search, or stacker fitting. Development decisions may look at 
chat_style_eval_v2.csv. VALIDATION was split into a stacking half and a 
selection half before any meta-learner or band-edge search.

## Checkpoints

- LSTM present: `True`
- DistilBERT present: `True`
- Skip note: three-model cascade ran

## TF-IDF two-threshold edges (VAL selection split)

- low (confident ham): `0.3`
- high (confident scam): `0.7`

## Confusion matrices ([[TN, FP], [FN, TP]])

| Arm | Set | Matrix |
| --- | --- | --- |
| TF-IDF single threshold | in-domain TEST | `[[3174, 489], [27, 3447]]` |
| TF-IDF single threshold | chat_style_eval_v2 | `[[24, 76], [0, 100]]` |
| TF-IDF band-as-legitimate | chat_style_eval_v2 | `[[97, 3], [61, 39]]` |
| TF-IDF single threshold | v1 locked predict-only | `[[29, 71], [0, 100]]` |
| TF-IDF band-as-legitimate | v1 locked predict-only | `[[98, 2], [43, 57]]` |
| LSTM single threshold 0.30 | in-domain TEST | `[[3463, 200], [49, 3425]]` |
| DistilBERT single threshold 0.30 | in-domain TEST | `[[3597, 66], [60, 3414]]` |
| Three-model cascade | in-domain TEST | `[[3557, 106], [67, 3407]]` |
| Two-tier TF-IDF→LSTM | in-domain TEST | `[[3512, 151], [60, 3414]]` |
| Soft-vote | in-domain TEST | `[[3492, 171], [22, 3452]]` |
| Stacker | in-domain TEST | `[[3574, 89], [46, 3428]]` |
| Three-model cascade | chat_style_eval_v2 | `[[84, 16], [18, 82]]` |
| Three-model cascade | v1 locked predict-only | `[[88, 12], [17, 83]]` |

A cascade that escalates to DistilBERT still requires shipping ~64 MiB. 
The two-tier TF-IDF→BiLSTM path is the shippable alternative.

## Length-mismatch experiment (max_chars=200)

Retrain artifacts live under `reports/length_filtered/`. That run kept 
41623/71370 rows, froze C=1.0 and threshold=0.30 on VALIDATION, and 
never fit or retuned on chat_style_eval_v1.csv.

| Model | Set | [[TN, FP], [FN, TP]] | ham warned |
| --- | --- | --- | ---: |
| Published TF-IDF (unfiltered) | v1 locked | `[[30, 70], [0, 100]]` | 70/100 |
| Length-filtered TF-IDF | v1 locked | `[[55, 45], [4, 96]]` | 45/100 |
| Length-filtered TF-IDF | v2 development | `[[56, 44], [2, 98]]` | 44/100 |
| Unfiltered TF-IDF (this run) | v1 locked | see table above | 71/100 |
| Unfiltered TF-IDF band-as-legitimate | v1 locked | see table above | 2/100 |

Length-filtering cuts locked-v1 false alarms from 70 to 45 per 100 ham 
messages and misses 4 scams (was 0). The band-as-legitimate two-threshold 
policy cuts false alarms to 2/100 on locked v1 but misses 43 scams — that 
is a policy choice, not a free lunch.

## Inference latency (Python, chat-eval v1+v2 texts)

From `reports/cascade/benchmark.json`. DistilBERT n=40 (first-call warmup
dominates p95); TF-IDF and LSTM n=400.

| Arm | p50 ms | p95 ms | p99 ms | ship cost |
| --- | ---: | ---: | ---: | --- |
| TF-IDF | 1.0 | 2.1 | 8.0 | tens of KiB ONNX head |
| Word BiLSTM | 2.4 | 4.2 | 5.2 | ~13.2 MiB |
| DistilBERT | 6.0 | 369 (warmup) | 369 | ~64.3 MiB |

On in-domain TEST the cascade settled 87.6% of rows at TF-IDF, 11.1% at
LSTM, and 1.3% at DistilBERT. A default that ships DistilBERT for 1.3% of
messages still has to download 64 MiB; two-tier TF-IDF→LSTM is the
browser-default candidate.
