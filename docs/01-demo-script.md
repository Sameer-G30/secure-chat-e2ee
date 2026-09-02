# Secure Chat demo script (Slice 9)

This is the reviewer walkthrough for a **local** Compose + Vite setup. Hosted Render/Railway/Fly.io deploy is **Slice 10**, not this slice.

A filmed recording was **not** captured in this environment (WSL2 agent session, no interactive two-browser recording). Follow the steps below on a machine with Docker Desktop and a Chromium-based browser. Do not commit a large binary in lieu of this script.

## 0. Start the stack (no manual Alembic)

From the repository root:

```bash
cp .env.example .env   # only if .env does not already exist; then replace placeholders
docker compose up --build
```

Wait until the `api` service is **healthy**. The API container runs `alembic upgrade head` on startup (including `conversations.last_rotated_at`, revision `a9f3c6e12b80`), then starts uvicorn. You should **not** run `docker compose run --rm api uv run --no-sync alembic upgrade head`.

In a second terminal:

```bash
cd frontend
npm install            # first time only
npm run dev
```

Use the origin Vite prints (usually `http://localhost:5173`). If it prints `5174`, or you open `127.0.0.1` instead of `localhost`, set `FRONTEND_ORIGIN` in `.env` to that exact origin and recreate the API container. CORS is a **single** origin with **GET/POST** only.

Confirm `GET http://localhost:8000/health` returns `"status": "ok"` and `"version": "0.9.0"`.

## 1. Two-tab E2EE (Slice 4 proof, still the core demo)

1. Open the Vite URL in **two** browser tabs.
2. Tab A: register `alice` (distinct handle), log in, wait for the chat shell (key upload finished).
3. Tab B: register `bob`, log in the same way.
4. Alice: **Add contact** `bob`. Bob: add `alice`. Contacts are server-side (`contacts` table), not `localStorage`.
5. Send a normal message one way, then the other. Each tab must show the recovered **plaintext** in the bubble.

Tokens stay in memory only: refreshing a tab requires logging in again. A new browser/device still hits `IdentitySetupError` (multi-device recovery is unsupported).

## 2. Network panel: ciphertext only

In DevTools → **Network** → the conversation **WebSocket**:

- Envelope frames must contain `ciphertext`, `nonce`, and `key_epoch`.
- They must **not** contain the message text, a `plaintext` field, or a private/session key.
- Typing frames are `{type: "typing", user_id, is_typing}` only — **no draft text**.
- Presence frames are `{type: "presence", user_id, online}`.
- Epoch frames (when a bump happens) are `{type: "epoch", current_epoch: <int>}` — the integer only, never a key.

Reload is optional for this step. Logging out and back in on the **same** browser, then opening the same contact, loads **conversation-scoped** history via `GET /conversations/{id}/messages`. That JSON is ciphertext-only (`ciphertext`, `nonce`, `key_epoch`, `sender_id`, `created_at`, `id`). This tab decrypts locally. There is no flat “all messages” query.

Optional DB check (BYTEA only):

```bash
docker compose exec postgres psql -U secure_chat -d secure_chat -c \
  "SELECT id, conversation_id, sender_id, octet_length(ciphertext), octet_length(nonce), key_epoch FROM messages;"
```

There is no `plaintext` / `body` / `private_key` column.

## 3. Optional epoch bump (Slice 8; default is 50 messages)

Production default: increment `conversations.current_epoch` after **50** persisted messages since the last bump **or 24 hours** since `last_rotated_at` (NULL treated as `created_at`). There is **no** unauthenticated rotate endpoint.

To demo a bump in two tabs:

1. Set `EPOCH_ROTATE_AFTER_MESSAGES=2` in `.env` (leave `EPOCH_ROTATE_AFTER_HOURS=24`).
2. Recreate the API container (`docker compose up -d --force-recreate api` is enough; no extra migrate command).
3. Send two messages in the conversation. After the second persist, both tabs should show status **epoch 1**, and the **next** send’s WebSocket frame should have `key_epoch: 1`.
4. The composer **draft is not cleared** by the bump.
5. Log out and back in on the same browser, open the same contact: the first two history rows still decrypt (`key_epoch: 0`). History GET remains ciphertext-only.

Automated proof of the same relay path: `cd backend && uv run pytest tests/test_epoch_rotation.py` (tests set N=2; they do not change the production default).

Epoch rotation is **not** forward secrecy. Compromising the static master session key still derives every epoch.

## 4. Scam banner on verified plaintext (Slice 6)

ChatScreen eagerly classifies **verified** plaintext with **TF-IDF Best** (10k terms, C=1.0, sidecar threshold 0.20, ChatScreen overlay 0.35) in the browser after decrypt (and on send, on plaintext the sender already has). Scoring uses the last 6 verified turns; trivial short lines with no URL are not classified. Scores never leave the tab.

Send a locked-chat-style scam DM (prize, seed phrase, fake support). A non-blocking **This message shows signs of a scam** banner may appear on a verified bubble. The message is never hidden, blocked, or deleted.

Optional: check **Use DistilBERT (large download)** or **Use Word BiLSTM Best** (XOR; one heavy graph at a time). DistilBERT is unpadded 256-token int8 (`distilbert_default`, threshold 0.30) in an ORT worker with package-export `wasmPaths`; the tab fetches int8 `model.onnx` (gzip/brotli), not `model.fp32.onnx`. Last banner flags are cached locally so reload can paint warnings before WASM is ready. Do not treat the six-row ORT table as TEST accuracy; see `ml/reports/onnx_web_load_check.md`.

## 5. Verification-failed row

If an envelope fails AEAD verification (tampered ciphertext or wrong associated data), the row must show **message failed verification**. It must **not** invent plaintext and must **not** show a scam banner.

## 6. What this demo is not

- Not a Double Ratchet and not true forward secrecy.
- Not a hosted production deploy (Slice 10).
- Not live URL reputation (no HTTP fetch, no Safe Browsing / VirusTotal / PhishTank).
- Not multi-device key recovery, httpOnly cookies, or `localStorage` tokens.
- Frontend is still `npm run dev` on the host; it is **not** a Compose/Nginx service in this slice.
