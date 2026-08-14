# Secure Chat with Client-Side Scam Detection

Secure Chat is a portfolio-grade real-time messaging system designed so the server stores and relays ciphertext but never receives message plaintext or private/symmetric key material. Decrypted messages are classified for phishing and scam indicators locally in the recipient's browser.

> Status: Slice 4. Registration and login are real (Argon2id + Postgres + rate limiting), JWT access/refresh tokens rotate on use with reuse detection, X25519 public keys can be uploaded/looked up over the authenticated API, and the browser generates its own identity keypair and seals the private half in IndexedDB with Argon2id. Two browser tabs can now hold a real end-to-end encrypted 1:1 conversation: the server stores and relays `{ciphertext, nonce, key_epoch}` only, and clients encrypt with XChaCha20-Poly1305 plus associated data before send. Client-side scam classification still arrives in later reviewed slices.

## Architecture

The browser generates its X25519 identity keypair locally (`crypto/keyExchange.ts`), seals the private half in IndexedDB with a password-derived Argon2id key (`crypto/keyVault.ts`), derives directional session/epoch keys, encrypts with XChaCha20-Poly1305, and decrypts authenticated envelopes. FastAPI authenticates users, issues/rotates JWTs, stores/serves public keys, coordinates the non-secret per-conversation epoch counter, and relays conversation-scoped ciphertext over WebSocket. PostgreSQL stores account metadata (`users`), refresh-token hashes (`refresh_tokens`), 1:1 membership plus `current_epoch` (`conversations`), and encrypted envelopes only (`messages`: ciphertext, nonce, key_epoch). The server has no code path that decrypts a message or handles a private/symmetric key.

## Threat model



### Intended protections

- Database dumps and honest-but-curious server operators cannot read message content because decryption keys remain on end-user devices.
- XChaCha20-Poly1305 authenticates ciphertext and associated conversation metadata, making tampering and cross-conversation replay detectable.
- Argon2id password hashing, short-lived (15 min) access tokens, rotating refresh tokens (single-use, 7-day lifetime), and rate limits on both `/auth/register` and `/auth/login` reduce account-compromise risk.
- Refresh-token rotation detects reuse: presenting an already-rotated token revokes every other active refresh token for that account, forcing full re-authentication instead of silently tolerating a stolen token.
- The X25519 private key is generated in the browser, never transmitted, and only ever persisted sealed (AEAD-encrypted with an Argon2id-derived, password-based key) in IndexedDB; the server only ever receives and stores the public half.
- React text rendering avoids raw HTML injection from user-controlled messages and usernames.



### Explicit limitations

- The server can observe metadata such as account identities, conversation membership, message timing, message size, and non-secret epoch numbers.
- A compromised endpoint, malicious browser extension, or attacker who knows the user's password and accesses the local key vault can read plaintext available on that endpoint.
- Static long-term X25519 keys plus epoch derivation provide key separation, not true forward secrecy. A complete Double Ratchet is intentionally out of scope.
- Public-key authenticity is initially trusted on first use; key transparency or out-of-band safety-number verification is future work.
- Multi-device key synchronization and recovery are outside the first implementation scope.
- Browser WebSocket clients cannot set an `Authorization` header, so Slice 4 sends the short-lived access token as a query parameter (`?access_token=`). Access logs or a copied URL could capture that token until it expires (15 minutes). A one-time WebSocket ticket is future hardening; do not treat the query string as a long-lived secret.



## Security design decisions

- **AEAD:** use `crypto_aead_xchacha20poly1305_ietf_`*, not `crypto_secretbox`, so the mandated XChaCha20 primitive and associated data are both supported.
- **Associated data:** authenticate a canonical encoding of `conversation_id`, `sender_id`, and `key_epoch`.
- **KDF context:** use the required eight-byte context `msgkey01`; the original seven-byte value would fail in libsodium.
- **Epoch claims:** describe the design as per-epoch key separation and compromise containment, not forward secrecy.
- **JWT library:** use maintained PyJWT instead of `python-jose`.
- **Baseline ONNX:** perform TF-IDF in TypeScript and export only the numerical classifier head to avoid unsupported ONNX Runtime Web tokenizer operators.
- **DistilBERT:** lazy-load the optional quantized model while eagerly loading the smaller baseline classifier.
- **`users.public_key` stays nullable — a documented transitional rule, not an oversight.** `POST /keys/me` requires a bearer access token, but `POST /auth/register` intentionally returns no tokens. Key upload therefore happens on the client's first successful `POST /auth/login`. Adding a database `NOT NULL` constraint would force bundling key upload into registration or inventing a placeholder key. Slice 4 conversation and message endpoints are the enforcement point: they reject either party whose `public_key is None` before a conversation can start or an envelope can be relayed.
- **JWT design (A4, PyJWT).** Access tokens are short-lived (15 min) and carry `sub`/`username`/`type`/`exp`; they are never persisted server-side. Refresh tokens are also JWTs (self-verifying signature/expiry) but the server additionally stores only a SHA-256 hash of each issued refresh token in `refresh_tokens`, matching Part B's schema refinement (`created_at`, `revoked_at`, `UNIQUE(token_hash)`) — a database read can never be turned into a usable token, the same principle as never storing recoverable passwords.
- **Refresh rotation with reuse detection.** Every `POST /auth/refresh` call revokes the presented token, win or lose. If a request presents a token that was *already* revoked (i.e., already rotated once, or logged out), that is treated as evidence of theft/replay: every other active refresh token for that account is revoked immediately, forcing the legitimate user to log in again everywhere. This goes beyond the spec's literal "rotated on use" requirement because rotation alone does not detect a stolen-and-replayed old token; verified by `backend/tests/test_auth_refresh.py`.
- **Login never distinguishes "no such user" from "wrong password."** Both return the identical `401 invalid username or password`, so the endpoint cannot be used to enumerate registered usernames — an explicit anti-pattern the spec singles out for hash/secret comparisons, and the same principle applies to any account-existence oracle.
- **`GET /keys/{username}` requires authentication even though public keys are "not secret data" (§6.2).** Gating it behind a valid access token prevents unauthenticated account-username enumeration via the key-lookup endpoint, at essentially no cost to legitimate use (a client already has to log in before it needs any peer's key).
- **Client-side key vault uses `libsodium-wrappers-sumo`, loaded on demand.** `crypto/keyExchange.ts` deliberately stays on the smaller `libsodium-wrappers` build (crypto_kx/crypto_kdf/AEAD only) so it can load eagerly on first paint. `crypto/keyVault.ts` needs `crypto_pwhash` (Argon2id) for password-derived vault sealing, which only the larger "sumo" build provides; it is dynamically `import()`ed so its extra ~190 kB (gzipped) WASM payload is fetched only when a login/registration actually seals or unseals a local identity key — the same lazy-loading principle as A6's DistilBERT opt-in.
- **`crypto_pwhash` uses `OPSLIMIT_INTERACTIVE`/`MEMLIMIT_INTERACTIVE`, not `MODERATE` or `SENSITIVE`.** These are libsodium's parameters tuned for immediate, in-browser, foreground use; `MODERATE`/`SENSITIVE` are meant for background/server contexts and would make every login noticeably slow in a tab. This is a documented security/UX trade-off: INTERACTIVE limits are weaker against offline brute-forcing of a stolen IndexedDB export than SENSITIVE limits would be, but the private key they protect is also independently useless without the account password, and IndexedDB exfiltration already requires a compromised endpoint (see the threat model above).
- **In-memory-only tokens and session state.** `AuthContext` holds the access token, refresh token, and unsealed private key only in React state — never in `localStorage`/`sessionStorage`/cookies. Reloading the page always requires logging in again. httpOnly-cookie-based refresh-token persistence is noted as future hardening.
- **WebSocket ciphertext relay (Slice 4).** `POST /conversations` starts or fetches the unique 1:1 row for a pair (`UNIQUE (user_a_id, user_b_id)` plus `CHECK (user_a_id < user_b_id)`). `GET /conversations/{id}/epoch` and the spec alias `GET /keys/conversations/{id}/epoch` return the non-secret `current_epoch` integer. `WS /ws/conversations/{id}` accepts `{ciphertext, nonce, key_epoch}`, persists BYTEA columns only, and fans the envelope to the peer. Optional `conversation_id` / `sender_id` on the frame are compared to the authenticated path/user and rejected on mismatch; cryptographic verify of associated data (`conversation_id`, `sender_id`, `key_epoch`) stays client-side (A1). Every messages-table query is scoped by `conversation_id`.
- **Epoch is key separation, not forward secrecy.** Slice 4 reads `current_epoch` (default 0) and passes it into `crypto_kdf_derive_from_key` with context `msgkey01`. Scheduled rotation is not implemented yet. Compromising the static master session key still derives every epoch; this is compromise containment, not a Double Ratchet.
- **Multi-device key recovery is explicitly unsupported, not silently broken.** If an account already has a public key on the server but the current browser has no matching sealed vault entry (a new device, a cleared profile, or a different browser), login intentionally fails with a clear `IdentitySetupError` rather than generating a second, unlinked identity that could never decrypt messages sent to the account's real public key. This matches the threat model's existing "multi-device key synchronization and recovery are outside the first implementation scope" limitation.



## Repository layout

- `backend/` — FastAPI application, tests, Docker image, and Alembic migrations.
- `frontend/` — React, Vite, TypeScript, crypto proof, component tests, and later browser inference.
- `ml/` — offline data preparation, EDA, model training, evaluation, and export.
- `docs/` — literature review and later architecture/security documentation.
- `Legacy files/` — insecure prototype retained locally for visual reference and intentionally excluded from Git.



## Local setup



### Prerequisites

- WSL2 Ubuntu
- Docker Desktop with WSL integration enabled
- Python 3.11.9 managed by pyenv
- `uv`
- Node.js 22 managed by nvm



### Environment

```bash
cp .env.example .env
```

Replace placeholder passwords and signing keys in `.env`. Never commit that file.

### Backend without Docker

```bash
cd backend
uv sync
uv run pytest
uv run uvicorn app.main:app --reload
```

The health endpoint is `http://localhost:8000/health`. Point `DATABASE_URL` at a reachable Postgres instance (e.g. `postgresql+asyncpg://secure_chat:<password>@localhost:5432/secure_chat` if you expose the Compose Postgres port locally) before running the app outside Docker; the automated tests do not need this because they run against an isolated in-memory SQLite database instead.

### Database migrations

```bash
cd backend
uv run alembic upgrade head   # apply migrations against DATABASE_URL
uv run alembic downgrade base # roll back everything (local development only)
```

Inside Docker Compose, run the same commands from the `api` container: `docker compose run --rm api uv run --no-sync alembic upgrade head`. The API container does not run migrations automatically on startup in this slice; run them once after `docker compose up` against a fresh database.

### Full local stack

```bash
docker compose up --build # --build only when image needs rebuilding or when you changed something that goes into docker image, otherwise do only docker compose up
docker compose run --rm api uv run --no-sync alembic upgrade head # first time, or after a new migration
```

Docker Compose starts FastAPI and PostgreSQL. The database has no host port because only the API should access it. Try the full auth + key flow once migrated:

```bash
# Register (no tokens issued; matches the tested Slice 2 registration contract).
curl -X POST http://localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo_user","email":"demo@example.com","password":"correct horse battery staple"}'

# Log in to get an access/refresh token pair; has_public_key is false on a fresh account.
curl -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo_user","password":"correct horse battery staple"}'

# Upload a (here: placeholder) base64 X25519 public key with the access token from above.
curl -X POST http://localhost:8000/keys/me \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer <access_token>' \
  -d '{"public_key":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}'

# Look up any account's public key (authenticated; the key itself is not secret).
curl http://localhost:8000/keys/demo_user -H 'Authorization: Bearer <access_token>'

# Start or fetch a 1:1 conversation with a peer who also has a public key.
curl -X POST http://localhost:8000/conversations \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer <access_token>' \
  -d '{"peer_username":"other_user"}'

# Read the non-secret epoch counter (spec §6.4 path; GET /conversations/{id}/epoch is equivalent).
curl http://localhost:8000/keys/conversations/<conversation_id>/epoch \
  -H 'Authorization: Bearer <access_token>'

# Rotate the refresh token; the presented token is revoked whether or not this succeeds.
curl -X POST http://localhost:8000/auth/refresh \
  -H 'Content-Type: application/json' -d '{"refresh_token":"<refresh_token>"}'
```

### Frontend

```bash
cd frontend
npm install # first time only
cp .env.example .env # only needed if the API is not on http://localhost:8000
npm run test
npm run dev
```

Open `http://localhost:5173`. Register an account, then log in: the first successful login generates an X25519 keypair in the browser, seals the private half in IndexedDB (Argon2id-derived key), uploads the public half via `POST /keys/me`, and lands on `ChatScreen`. Logging out and back in on the same browser unseals the same identity from IndexedDB instead of generating a new one.

### Two-tab encrypted conversation (Slice 4 proof)

Both API and frontend must be running, and migrations must include `conversations` and `messages` (`alembic upgrade head` as shown above).

1. Open `http://localhost:5173` in **two browser tabs**.
2. Tab A: register `alice` (or any distinct handle), then log in. Wait until the chat shell appears — that means key upload finished.
3. Tab B: register `bob`, then log in the same way.
4. In Alice's tab, enter `bob` as the peer username and click **Start encrypted chat**. In Bob's tab, enter `alice` and start the chat. Either order is fine; the server stores one canonical row with `user_a_id < user_b_id`.
5. Type a message in one tab and send it. The other tab should show the recovered plaintext. The Network panel's WebSocket frames must contain `ciphertext`, `nonce`, and `key_epoch` — never the message text.
6. If you tamper with a frame (or the associated conversation/sender metadata), the recipient shows **message failed verification** instead of garbled text.

Tokens stay in memory only: refreshing a tab requires logging in again. Multi-device key recovery is still unsupported (`IdentitySetupError`). Do not expect contacts, history pagination, typing/presence, dark mode, or a scam banner in this slice.

To inspect ciphertext-only storage after a send (API container + `psql` are optional; the automated tests already assert this):

```bash
# After sending a message in the two-tab UI, the messages table holds BYTEA only.
docker compose exec postgres psql -U secure_chat -d secure_chat -c \
  "SELECT id, conversation_id, sender_id, octet_length(ciphertext), octet_length(nonce), key_epoch FROM messages;"
```

There is no `plaintext` / `body` / `private_key` column. `GET /keys/conversations/{id}/epoch` (or `GET /conversations/{id}/epoch`) returns only `{conversation_id, current_epoch}`.

### ML workspace

```bash
cd ml
uv sync
uv run python scripts/download_sms_spam.py
uv run python scripts/download_enron_spam.py
uv run python scripts/download_spamassassin.py
uv run python scripts/download_nazario.py
uv run python scripts/download_kaggle_phishing.py  # requires ml/data/raw/Phishing_Email.csv
uv run jupyter lab
```

Raw downloaded datasets and generated model weights remain untracked. Open `notebooks/01_eda.ipynb` for SMS-only EDA or `notebooks/02_eda_all_corpora.ipynb` for the multi-corpus report.

### Training the TF-IDF baseline

```bash
cd ml
uv run python scripts/train_baseline.py
```

This combines every downloaded corpus in `data/processed/`, deduplicates and drops empty rows, takes a stratified 80/20 train/test split (seeded, `random_state=42`), fits `TfidfVectorizer` (unigrams+bigrams, `sublinear_tf`) → `LogisticRegression` (`class_weight="balanced"`), and writes `reports/baseline_metrics.json` plus `reports/confusion_matrix.png`. `uv run pytest` exercises the same pipeline against tiny synthetic data so CI never needs the multi-gigabyte raw corpora.

### Chat-style out-of-domain evaluation (Slice 3)

```bash
cd ml
uv run python scripts/build_chat_style_eval_set.py    # writes data/chat_eval/chat_style_eval_v1.csv
uv run python scripts/evaluate_chat_style_eval.py      # writes reports/chat_style_eval_metrics.json
```

`build_chat_style_eval_set.py` writes out 80 hand-authored, chat-register messages (40 ordinary DM-style texts, 40 scam/phishing DM-style texts covering romance, crypto, prize, "hi mom/it's me," fake-support, and phishing-link patterns) — none of it scraped, all of it written or reviewed for this project. Per `data/label-schema.yaml`'s `evaluation_policy`, `evaluate_chat_style_eval.py` only ever calls `.predict()` on this set; it fits the pipeline once on the full in-domain corpus first.

## Testing

- Backend: `cd backend && uv run pytest`
- Backend lint: `cd backend && uv run ruff check .`
- Backend types: `cd backend && uv run mypy app tests`
- Frontend tests: `cd frontend && npm run test`
- Frontend build: `cd frontend && npm run build`
- Frontend lint: `cd frontend && npm run lint`
- ML unit tests (synthetic data, no download required): `cd ml && uv run pytest`
- ML lint: `cd ml && uv run ruff check scripts src tests`
- Full service health: `docker compose up --build`, migrate, then exercise `/health`, `/auth/register`, `/auth/login`, `/keys/me`, `/keys/{username}`, `/auth/refresh`, `POST /conversations`, `GET /conversations/{id}/epoch` (or `GET /keys/conversations/{id}/epoch`), and a two-tab WebSocket conversation as shown above

Slice 1 proved the API health contract, a complete libsodium key-exchange/KDF/AEAD round trip, rejection of tampered ciphertext, and a production frontend build. Slice 2 added: a migrated `users` table; Argon2id-hashed, rate-limited, uniqueness-enforced registration exercised end-to-end through Docker Compose against real Postgres; the crypto spike productionized into `keyExchange.ts`; `AuthScreen` wired to real registration; and a trained, evaluated TF-IDF baseline.

Slice 3 added:

- **Backend:** `POST /auth/login` (Argon2id verify, rate-limited); PyJWT access tokens (15 min) and single-use, rotate-on-refresh refresh tokens with theft/replay detection (`POST /auth/refresh`); `POST /auth/logout`; a migrated `refresh_tokens` table (hash-only, `created_at`/`revoked_at`/`UNIQUE(token_hash)`); authenticated `POST /keys/me` and `GET /keys/{username}`; 19 new backend tests covering login, rotation, reuse detection, key upload/lookup validation, and rate limits.
- **Frontend:** `AuthScreen` wired to real login; a new `crypto/keyVault.ts` (IndexedDB, Argon2id-sealed private key); `crypto/identitySetup.ts` reconciling server/local key state on every login; an in-memory-only `AuthContext` (no tokens in `localStorage`); a minimal protected `ChatScreen` shown once a session exists.
- **ML:** the hand-curated, evaluation-only chat-style set (`ml/data/chat_eval/chat_style_eval_v1.csv`, 80 rows) and its out-of-domain evaluation report, discussed below.

Slice 4 adds:

- **Backend:** migrated `conversations` (`UNIQUE (user_a_id, user_b_id)`, `CHECK (user_a_id < user_b_id)`, `current_epoch` default 0) and `messages` (ciphertext/nonce BYTEA, `key_epoch`, index on `(conversation_id, created_at DESC)`); `POST /conversations` start-or-fetch; `GET /conversations/{id}` membership-gated fetch; `GET /conversations/{id}/epoch` plus spec alias `GET /keys/conversations/{id}/epoch`; authenticated `WS /ws/conversations/{id}` ciphertext relay that never decrypts; rejection of missing public keys, non-members, spoofed `sender_id`, cross-conversation `conversation_id`, and future `key_epoch`; 48 backend tests including a grep/AST sweep that the server never names decrypt/private-key/plaintext identifiers.
- **Frontend:** `ChatScreen` beyond the placeholder — peer username, `GET /keys/{username}`, `deriveSessionKeys` / lexicographic `crypto_kx` roles, epoch fetch, WebSocket send/receive, XChaCha20-Poly1305 + associated data, and a **message failed verification** state that never renders corrupted plaintext.
- **ML:** unchanged from Slice 3. DistilBERT / ONNX Runtime Web are still later slices. The chat-style eval numbers below still apply.

## E2EE and client-side AI

The sender encrypts before network transmission. The server stores and relays only an authenticated encrypted envelope. The recipient verifies and decrypts locally. Scam classification on the recovered plaintext in ONNX Runtime Web is still a later slice; Slice 4 stops at verified plaintext in the message list.

## ML evaluation

The baseline and DistilBERT tracks report precision, recall, F1, confusion matrices, selected thresholds, model size, and browser latency. Accuracy alone is insufficient for the imbalanced and harm-sensitive classification task.

### TF-IDF + Logistic Regression baseline (Slice 2)

Trained on 50,793 rows and evaluated on a held-out stratified test split of 12,699 rows, combining UCI SMS Spam, Enron-Spam, SpamAssassin, Nazario phishing, and the Kaggle phishing-email compilation (`ml/reports/baseline_metrics.json`, regenerate with `uv run python scripts/train_baseline.py`):

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| legitimate | 0.979 | 0.974 | 0.977 |
| scam | 0.968 | 0.974 | 0.971 |

![Baseline confusion matrix](ml/reports/confusion_matrix.png)

**Threshold:** the default `predict_proba >= 0.5` decision boundary from `LogisticRegression`, with `class_weight="balanced"` compensating for the roughly 55/45 legitimate/scam split rather than a hand-tuned threshold shift.

**Precision/recall trade-off reasoning:** false negatives (a scam shown with no warning) and false positives (a legitimate message flagged) are both costly, but not symmetrically — a missed scam can cause real harm, while an over-flagged legitimate message only costs the recipient a moment's doubt because the spec requires the warning to be non-blocking (§7: never hide, block, or auto-delete the message). At the default threshold, recall on the scam class (0.974) is already slightly higher than precision (0.968), which is the right side to lean toward for a "warn, don't block" UX. We keep the default threshold for this baseline rather than shifting it further toward recall, because that would sacrifice more legitimate-message precision than the current false-positive rate (183 of 7,013 legitimate test rows) justifies; this will be revisited once the hand-curated chat-style evaluation set (Slice 3+) shows how the threshold behaves outside email/SMS-register text.

**Known limitation, stated plainly:** every row the baseline was *trained* on comes from email and SMS corpora, not real chat messages. The out-of-domain check below exists precisely because that limitation is real, not hypothetical.

### Out-of-domain check: the hand-curated chat-style eval set (Slice 3)

`data/label-schema.yaml`'s `evaluation_policy` reserves this 80-row hand-authored set (`ml/data/chat_eval/chat_style_eval_v1.csv`, built by `scripts/build_chat_style_eval_set.py`) exclusively for evaluation; `scripts/evaluate_chat_style_eval.py` fits the same pipeline on the full in-domain corpus and only ever calls `.predict()` on this set (`ml/reports/chat_style_eval_metrics.json`):

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| legitimate | 0.830 | 0.975 | 0.897 |
| scam | 0.970 | 0.800 | 0.877 |

Confusion matrix (rows = true, columns = predicted, order `[legitimate, scam]`): `[[39, 1], [8, 32]]` — 1 false positive out of 40 ordinary chat messages, but 8 missed scams out of 40 hand-authored scam-style chat messages.

**This confirms the register-shift risk the Slice 2 README already flagged in advance:** scam recall drops from 0.974 in-domain to 0.800 out-of-domain — the model misses 1 in 5 chat-style scams, mostly ones that lack the email/SMS corpora's characteristic boilerplate ("verify your account," "click here," dollar amounts with urgency framing) and instead use short, casual phrasing ("hey it's me, I lost my phone, can you send money"). Precision stays high (0.970): when the baseline does flag a chat message as a scam, it is very rarely wrong.

**Threshold selection, revisited with this data:** the spec's "warn, don't block" UX (§7) means a false positive costs a moment of doubt while a false negative gives zero warning at all — the two error types are not symmetric, and recall matters more. We are **not** shifting the baseline's `predict_proba >= 0.5` threshold down in Slice 3, for two reasons: (1) with only 80 hand-labeled rows, tuning a threshold against this exact set would overfit to our own 40 examples rather than generalize; (2) the deeper fix for a register mismatch is more representative training data or a model better suited to short informal text (DistilBERT, A6), not a threshold shift on the same TF-IDF features. This eval set's job is to *measure* the gap honestly, which it now does; closing it is explicit future work for the DistilBERT track and/or a larger chat-style corpus, not a Slice 3 deliverable.

## Deployment

Deployment is scheduled for later slices after authentication, conversation-scoped storage, E2EE relay, model integration, and security verification are complete.

## Demo

A short two-browser E2EE demonstration is the Slice 4 proof above. A scam-warning demonstration will be recorded in the final slices.