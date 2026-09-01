# Length and truncation audit

## DistilBERT tokenizer configuration (verbatim)

The serving and training call in `secure_chat_ml.distilbert.tokenize_texts` is:

```python
encoded = tokenizer(
    texts,
    truncation=True,
    max_length=256,
    padding=False,
)
```

Model: `distilbert-base-uncased` (WordPiece). TF-IDF has no length cap.

## Character-length percentiles

| Split | n | p50 | p90 | p95 | p99 | p100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 49958 | 183.0 | 320.0 | 385.0 | 577.4 | 40359.0 |
| val | 14275 | 183.0 | 320.0 | 379.0 | 565.5 | 19760.0 |
| test | 7137 | 184.0 | 322.0 | 395.0 | 591.6 | 48187.0 |
| chat_eval_v1 | 200 | 57.5 | 91.0 | 96.0 | 100.0 | 104.0 |

## DistilBERT TRAIN overflow from already-run reports

| max_length | truncated TRAIN rows | source |
| ---: | ---: | --- |
| 128 | 1787 | `reports/distilbert_param_sweep/07_max_length_128/test_metrics.json` |
| 256 | 297 | `reports/distilbert/test_metrics.json` |
| 384 | 225 | `reports/distilbert_param_sweep/08_max_length_384/test_metrics.json` |
| 512 | 196 | `reports/distilbert_param_sweep/09_max_length_512/test_metrics.json` |

WordPiece live source: `distilbert-base-uncased local_files_only`

## Live WordPiece percentiles

| Split | n | p50 | p90 | p95 | p99 | p100 | overflow@256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 49958 | 48.0 | 92.0 | 115.0 | 192.0 | 24817.0 | 297 |
| val | 14275 | 48.0 | 92.0 | 115.0 | 174.3 | 10811.0 | 66 |
| test | 7137 | 49.0 | 93.0 | 117.0 | 199.3 | 20659.0 | 46 |
| chat_eval_v1 | 200 | 15.0 | 25.0 | 26.0 | 29.0 | 29.0 | 0 |

## Chunking is rejected

Chunked (sliding-window) DistilBERT inference is rejected for this product. Only 297/49958 TRAIN rows overflow 256 WordPiece tokens (0.59%), and the locked chat-style eval set's p100 token length is 29.0 — well under the serving cap. Splitting a DM into windows would invent a second, untrained aggregation rule (max/mean/any-window-warn) on a domain that does not need it, and would multiply the 345 ms DistilBERT cost in the browser. The false-alarm problem is not truncation; it is the 0.85 in-domain legitimate-recall floor.

