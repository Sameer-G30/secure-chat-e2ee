# ONNX Runtime Web six-way load check (Slice 6)

This file records **browser** cost for the six exported checkpoints. It does **not** overwrite `baseline_metrics.json`, `val_metrics.json`, DistilBERT/LSTM `test_metrics.json`, or any sweep JSON. Fixture DMs are four short ham/scam messages (with and without URLs). They are **not** TEST accuracy and are **not** the locked 200-row chat-eval set.

Labels stay 0=legitimate, 1=scam. Thresholds were not retuned. `live_url_reputation` stays false.

Gzip/brotli and omitting `model.fp32.onnx` do **not** change weights. Precision / recall / accuracy below are the existing DistilBERT-best TEST and locked chat-eval reports.

## How this was measured

1. Export (does not write reports metric JSON):

   ```bash
   cd ml
   uv run python scripts/export_onnx_web.py
   ```

   Re-export one id: `uv run python scripts/export_onnx_web.py --only tfidf_default`.

   Refresh an existing `frontend/public/ml` tree without re-quantizing:

   ```bash
   uv run python scripts/prepare_browser_onnx.py
   ```

2. Serve the frontend (`cd frontend && npm run dev`) so Vite can fetch `/ml/<id>/`. Vite serves `model.onnx` with `Content-Encoding: br` or `gzip` when sibling `.br` / `.gz` files exist (`frontend/vite.precompressedOnnx.ts`).

3. Open `http://127.0.0.1:5173/?mlLoadCheck=1` (or `node frontend/scripts/ort_web_load_check.mjs` against a running Vite). Checkpoints load **one at a time**. DistilBERT and the word BiLSTM never share a WASM session.

Latest run: WSL2 Ubuntu, Playwright Chromium 151, Vite 8, COOP/COEP on. DistilBERT infers **unpadded** short fixtures (not pad-to-256/512). Raw JSON: `ml/reports/onnx_web_load_check.json`.

## Six-row browser table

| # | Checkpoint | Load | Init | Infer / msg (mean) | Uncompressed ONNX | Download (br) | Fixture banners vs Python | Offline TEST FN / FP | Offline chat FN / FP |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | DistilBERT best (max_length 512, thr 0.20) | yes (int8) | 916 ms | 29 ms (p50 19) | 64.3 MiB | 37.1 MiB | 4/4 match | 49 / 67 (acc 0.9837) | 8 / 6 (acc 0.930) |
| 2 | DistilBERT default (max_length 256, thr 0.30) | yes (int8) | 597 ms | 11 ms | 64.3 MiB | 37.1 MiB | 3/4 match | 60 / 66 (acc 0.9823) | 12 / 9 (acc 0.895) |
| 3 | Word BiLSTM best (8 epochs, thr 0.20) | yes (fp32) | 189 ms | 1.5 ms | 13.2 MiB | 12.1 MiB | 4/4 match | 30 / 179 (acc 0.9707) | 11 / 29 (acc 0.800) |
| 4 | Word BiLSTM default (4 epochs, thr 0.30) | yes (fp32) | 186 ms | 0.6 ms | 13.2 MiB | 12.1 MiB | 4/4 match | 49 / 200 (acc 0.9651) | 14 / 24 (acc 0.810) |
| 5 | TF-IDF best (10k, C=1.0, thr 0.20) | yes (fp32 head) | 19 ms | 0.8 ms | 39 KiB | 36 KiB | 4/4 match | 17 / 440 (acc 0.9360) | 0 / 61 (acc 0.695) |
| 6 | TF-IDF default (50k, C=0.25, thr 0.30) | yes (fp32 head) | 37 ms | 0.8 ms | 196 KiB | 178 KiB | 4/4 match | 27 / 489 (acc 0.9277) | 0 / 70 (acc 0.650) |

Init for row 1 includes a **cold** ORT WASM start plus brotli decode of 37.1 MiB. Row 2 is the same architecture after that warmup, so 916 vs 597 ms is mostly load order, not “512 is slower to load.” Short-fixture infer is no longer hundreds of milliseconds: the previous 759 / 345 ms/msg numbers were **padded** to `max_length`.

DistilBERT default 3/4 fixture banners: int8 vs PyTorch fp32 can flip a score near the 0.30 threshold. DistilBERT best (ChatScreen opt-in) matched 4/4.

## DistilBERT Best quality (unchanged by gzip / dropping fp32)

Same weights as `ml/reports/distilbert_param_sweep/09_max_length_512/`. Compression only changes transfer size.

**In-domain TEST** (7,137 rows, threshold 0.20):

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| legitimate | 0.987 | 0.982 | 0.984 |
| scam | 0.981 | 0.986 | 0.983 |
| accuracy |  |  | 0.984 |

Confusion `[[TN, FP], [FN, TP]]`: `[[3596, 67], [49, 3425]]`.

**Locked chat-eval v1** (200 rows; predict-only):

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| legitimate | 0.922 | 0.940 | 0.931 |
| scam | 0.939 | 0.920 | 0.929 |
| accuracy |  |  | 0.930 |

Confusion: `[[94, 6], [8, 92]]` — 6 ham warned, 8 scams missed. DistilBERT default on the same file was 9 ham / 12 missed (acc 0.895).

## Download sizes (DistilBERT int8 `model.onnx`)

| Encoding | DistilBERT best | DistilBERT default |
| --- | ---: | ---: |
| uncompressed (RAM after decode) | 67,404,393 B (64.3 MiB) | same |
| gzip | 41,898,605 B (40.0 MiB) | 41,904,492 B |
| brotli (what Chromium negotiated) | 38,948,206 B (37.1 MiB) | 38,943,502 B |

`model.fp32.onnx` (~256 MiB) stays under `ml/exports/onnx_web/` only. It is **not** copied to `frontend/public/ml/`. A request for that filename from Vite is the HTML app shell, not a 256 MiB graph.

## ChatScreen default

**Eager ChatScreen model: TF-IDF Best** (`tfidf_best`, 10k terms, C=1.0, sidecar threshold 0.20). ChatScreen then requires P(scam) ≥ 0.35 and scores the last 6 verified turns; `/?mlLoadCheck=1` still uses the sidecar cut on isolated fixtures.

**Lazy DistilBERT opt-in: DistilBERT default** (`distilbert_default`, 256-token int8, threshold 0.30) via “Use DistilBERT (large download)”.

Why this, after all six loaded successfully:

- DistilBERT 256 is the Slice 5 published graph. Uncompressed int8 is still ~64.3 MiB in WASM RAM. Too heavy to eager-load on every chat (A6), ~37.1 MiB on the wire with brotli, and **unpadded** short DMs are tens of milliseconds.
- DistilBERT 512 remains the offline sweep export (`distilbert_best`) for `/?mlLoadCheck=1`. Not the ChatScreen checkbox (threshold 0.20 was noisier on ordinary DMs in the tab).
- ChatScreen DistilBERT **does not pad** short DMs (dynamic sequence axis) and runs `session.run` in an ORT Web Worker, with extra WASM threads when the page is cross-origin isolated (Vite COOP/COEP). Gzip/brotli change **download** size, not infer FLOPs. Banner flags are cached per username so reload can paint warnings before WASM is ready.
- Word BiLSTM Best is a second **lazy opt-in** (“Use Word BiLSTM Best”). Only one heavy graph is resident at a time. The published 4-epoch / 0.30 LSTM remains the switch-back, not the ChatScreen toggle.
- TF-IDF Best is the A5 eager path: TypeScript TF-IDF + 39 KiB logistic ONNX. Same frozen C/threshold as `ml/reports/baseline_param_sweep/01_max_features_10000/`. The published 50k TF-IDF default stays exported as a switch-back; do not delete those artifacts.

Trainer defaults (`train_baseline.py` / `train_distilbert.py` / `train_lstm.py`) were **not** changed.

## Re-export

```bash
cd ml
uv run python scripts/export_onnx_web.py                 # all six, copy into frontend/public/ml/ (no fp32)
uv run python scripts/export_onnx_web.py --only tfidf_default
uv run python scripts/export_onnx_web.py --no-quantize    # DistilBERT fp32 serving graphs (still omitted from public/ml)
uv run python scripts/prepare_browser_onnx.py            # strip leftover fp32 from public/ml; write .gz/.br
```

Published TF-IDF has no `models/baseline/pipeline.joblib`. The exporter fits TRAIN-only into `models/baseline_onnx_export/` (reports untouched) and reuses that dump on later runs.

## Two-tab demo with banner

1. `docker compose up` and migrate; `cd frontend && npm run dev`.
2. Two tabs at `http://localhost:5173`. Register/login `alice` and `bob` as in the Slice 4 proof.
3. Start the encrypted chat both ways. Send a DM. Network/WebSocket frames are still `{ciphertext, nonce, key_epoch}` only.
4. Verified plaintext (sent locally and received after decrypt) may show **This message shows signs of a scam**. Verification-failed rows never get a banner. The banner does not hide, block, or delete the text. Scores never leave the tab.
5. Optional: check **Use DistilBERT (large download)** to lazy-load Slice 5 `distilbert_default` (256-token int8, unpadded sequences, Web Worker). Check **Use Word BiLSTM Best** instead to lazy-load the 8-epoch LSTM. Uncheck either to unload that heavy graph and return to TF-IDF Best.
6. Sequential measurement page: `http://localhost:5173/?mlLoadCheck=1`.
