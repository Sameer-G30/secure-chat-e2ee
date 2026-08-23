# ONNX Runtime Web six-way load check (Slice 6)

This file records **browser** cost for the six exported checkpoints. It does **not** overwrite `baseline_metrics.json`, `val_metrics.json`, DistilBERT/LSTM `test_metrics.json`, or any sweep JSON. Fixture DMs are four short ham/scam messages (with and without URLs). They are **not** TEST accuracy and are **not** the locked 200-row chat-eval set.

Labels stay 0=legitimate, 1=scam. Thresholds were not retuned. `live_url_reputation` stays false.

## How this was measured

1. Export (does not write reports metric JSON):

   ```bash
   cd ml
   uv run python scripts/export_onnx_web.py
   ```

   Re-export one id: `uv run python scripts/export_onnx_web.py --only tfidf_default`.

2. Serve the frontend (`cd frontend && npm run dev`) so Vite can fetch `/ml/<id>/`.

3. Open `http://127.0.0.1:5173/?mlLoadCheck=1` (or `node frontend/scripts/ort_web_load_check.mjs` against a running Vite). Checkpoints load **one at a time** in onnxruntime-web WASM (`numThreads=1`). Each session is disposed before the next id. DistilBERT and the word BiLSTM never share a WASM session.

Machine: WSL2 Ubuntu, Playwright Chromium 140, `performance.memory` JS heap ≈ 91.7 MiB after the sequence. Raw JSON: `ml/reports/onnx_web_load_check.json`.

## Six-row browser table

| # | Checkpoint | Load | Init | Infer / msg | Serving ONNX | Fixture banners vs Python | Offline TEST FN / FP | Offline chat FN / FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DistilBERT best (max_length 512, thr 0.20) | yes (int8) | 642 ms | 759 ms | 64.3 MiB | 4/4 match | 49 / 67 (acc 0.9837) | 8 / 6 (acc 0.930) |
| 2 | DistilBERT default (max_length 256, thr 0.30) | yes (int8) | 254 ms | 345 ms | 64.3 MiB | 4/4 match | 60 / 66 (acc 0.9823) | 12 / 9 (acc 0.895) |
| 3 | Word BiLSTM best (8 epochs, thr 0.20) | yes (fp32) | 76 ms | 1.3 ms | 13.2 MiB | 4/4 match | 30 / 179 (acc 0.9707) | 11 / 29 (acc 0.800) |
| 4 | Word BiLSTM default (4 epochs, thr 0.30) | yes (fp32) | 72 ms | 0.8 ms | 13.2 MiB | 4/4 match | 49 / 200 (acc 0.9651) | 14 / 24 (acc 0.810) |
| 5 | TF-IDF best (10k, C=1.0, thr 0.20) | yes (fp32 head) | 17 ms | 0.7 ms | 39 KiB (+ vocab JSON) | 4/4 match | 17 / 440 (acc 0.9360) | 0 / 61 (acc 0.695) |
| 6 | TF-IDF default (50k, C=0.25, thr 0.30) | yes (fp32 head) | 37 ms | 1.0 ms | 196 KiB (+ 2.6 MiB `tfidf.json`) | 4/4 match | 27 / 489 (acc 0.9277) | 0 / 70 (acc 0.650) |

Sidecars (not in the ONNX column): DistilBERT WordPiece vocab 322 KiB; LSTM `lstm_meta.json` 470 KiB; published TF-IDF `tfidf.json` 2.6 MiB; TF-IDF 10k vocab is smaller. DistilBERT also keeps `model.fp32.onnx` (~256 MiB) next to the serving int8 graph for fallback; the tab fetches `model.onnx` (int8).

## ChatScreen default

**Eager ChatScreen model: published TF-IDF default** (`tfidf_default`, 50k terms, C=0.25, threshold 0.30).

Why this, after all six loaded successfully:

- DistilBERT 512 is the offline quality winner but costs ~65 MiB WASM plus ~759 ms per short DM (padded to 512). Too heavy to eager-load on every chat (A6).
- DistilBERT 256 is the Slice 5 switch-back: still ~65 MiB and ~345 ms/msg. **Lazy opt-in** via the ChatScreen checkbox (“Use DistilBERT (large download)”), not the always-on default.
- Word BiLSTM is fast (~73 ms init, ~1 ms/msg, 13 MiB) and all fixtures matched, but this slice does **not** wire it into ChatScreen. If a later slice exports LSTM for chat, start with the published 4-epoch / 0.30 point, not 8-epoch / 0.20.
- Published TF-IDF is the A5 path: TypeScript TF-IDF + 196 KiB logistic ONNX, 37 ms init, ~1 ms/msg. Same frozen C/threshold as `ml/reports/baseline_metrics.json`. The 10k sweep winner is kept as a switch-back if the 50k vocab JSON is a poor fit on a smaller device; do not delete the 50k artifacts.

Trainer defaults (`train_baseline.py` / `train_distilbert.py` / `train_lstm.py`) were **not** changed.

## Re-export

```bash
cd ml
uv run python scripts/export_onnx_web.py                 # all six, copy into frontend/public/ml/
uv run python scripts/export_onnx_web.py --only tfidf_default
uv run python scripts/export_onnx_web.py --no-quantize    # DistilBERT fp32 serving graphs
```

Published TF-IDF has no `models/baseline/pipeline.joblib`. The exporter fits TRAIN-only into `models/baseline_onnx_export/` (reports untouched) and reuses that dump on later runs.

## Two-tab demo with banner

1. `docker compose up` and migrate; `cd frontend && npm run dev`.
2. Two tabs at `http://localhost:5173`. Register/login `alice` and `bob` as in the Slice 4 proof.
3. Start the encrypted chat both ways. Send a DM. Network/WebSocket frames are still `{ciphertext, nonce, key_epoch}` only.
4. Verified plaintext (sent locally and received after decrypt) may show **This message shows signs of a scam**. Verification-failed rows never get a banner. The banner does not hide, block, or delete the text. Scores never leave the tab.
5. Optional: check **Use DistilBERT (large download)** to lazy-load the Slice 5 256-token int8 graph. Uncheck to unload it and return to TF-IDF.
6. Sequential measurement page: `http://localhost:5173/?mlLoadCheck=1`.
