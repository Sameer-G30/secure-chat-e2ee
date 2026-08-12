# Secure Chat with Client-Side Scam Detection

Secure Chat is a portfolio-grade real-time messaging system designed so the server stores and relays ciphertext but never receives message plaintext or private/symmetric key material. Decrypted messages are classified for phishing and scam indicators locally in the recipient's browser.

> Status: Slice 2. Registration is real (Argon2id + Postgres + rate limiting). Login, JWT sessions, key upload/lookup, IndexedDB key vault, real-time messaging, and model inference in the browser arrive in later reviewed slices.

## Architecture

The browser will generate X25519 keypairs, derive directional session and epoch keys, encrypt with XChaCha20-Poly1305, decrypt authenticated envelopes, and run ONNX inference. FastAPI authenticates users and will relay conversation-scoped ciphertext. PostgreSQL stores account metadata (as of Slice 2: the `users` table) and will store public keys, conversations, refresh-token hashes, non-secret epochs, and encrypted envelopes only.

## Threat model



### Intended protections

- Database dumps and honest-but-curious server operators cannot read message content because decryption keys remain on end-user devices.
- XChaCha20-Poly1305 authenticates ciphertext and associated conversation metadata, making tampering and cross-conversation replay detectable.
- Argon2id password hashing, short-lived access tokens, rotating refresh tokens, and rate limits reduce account-compromise risk.
- React text rendering avoids raw HTML injection from user-controlled messages and usernames.



### Explicit limitations

- The server can observe metadata such as account identities, conversation membership, message timing, message size, and non-secret epoch numbers.
- A compromised endpoint, malicious browser extension, or attacker who knows the user's password and accesses the local key vault can read plaintext available on that endpoint.
- Static long-term X25519 keys plus epoch derivation provide key separation, not true forward secrecy. A complete Double Ratchet is intentionally out of scope.
- Public-key authenticity is initially trusted on first use; key transparency or out-of-band safety-number verification is future work.
- Multi-device key synchronization and recovery are outside the first implementation scope.



## Security design decisions

- **AEAD:** use `crypto_aead_xchacha20poly1305_ietf_`*, not `crypto_secretbox`, so the mandated XChaCha20 primitive and associated data are both supported.
- **Associated data:** authenticate a canonical encoding of `conversation_id`, `sender_id`, and `key_epoch`.
- **KDF context:** use the required eight-byte context `msgkey01`; the original seven-byte value would fail in libsodium.
- **Epoch claims:** describe the design as per-epoch key separation and compromise containment, not forward secrecy.
- **JWT library:** use maintained PyJWT instead of `python-jose`.
- **Baseline ONNX:** perform TF-IDF in TypeScript and export only the numerical classifier head to avoid unsupported ONNX Runtime Web tokenizer operators.
- **DistilBERT:** lazy-load the optional quantized model while eagerly loading the smaller baseline classifier.
- **`users.public_key` is nullable in the Slice 2 migration.** The spec's schema (§5) marks it `not null`, but the client only generates and uploads its X25519 public key through `POST /keys/me`, which is a Slice 3 deliverable. Making the column `not null` in Slice 2 would force registration to invent a placeholder key — worse than an honest nullable column today, tightened to `not null` by a Slice 3 migration once every account is required to have completed key upload. No account is usable for E2EE messaging until that key exists; this is purely about not overclaiming what a fresh Slice 2 registration provides.
- **Login is not implemented in Slice 2.** `AuthScreen` now performs real registration against the backend; the login form intentionally still only switches view state, since real login needs the JWT access/refresh flow that ships in Slice 3. This keeps the auth screen's wiring in step with what the backend can actually authenticate.



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

Docker Compose starts FastAPI and PostgreSQL. The database has no host port because only the API should access it. Try registration once migrated:

```bash
curl -X POST http://localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"demo_user","email":"demo@example.com","password":"correct horse battery staple"}'
```

### Frontend

```bash
cd frontend
npm install # first time only
cp .env.example .env # only needed if the API is not on http://localhost:8000
npm run test
npm run dev
```

Open `http://localhost:5173`. The auth screen's registration flow calls the real `POST /auth/register` endpoint; switch to "Create one" to try it against a running backend.

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

## Testing

- Backend: `cd backend && uv run pytest`
- Backend lint: `cd backend && uv run ruff check .`
- Backend types: `cd backend && uv run mypy app tests`
- Frontend tests: `cd frontend && npm run test`
- Frontend build: `cd frontend && npm run build`
- ML unit tests (synthetic data, no download required): `cd ml && uv run pytest`
- ML lint: `cd ml && uv run ruff check scripts src tests`
- Full service health: `docker compose up --build`, migrate, then request `/health` and `POST /auth/register`

Slice 1 proved the API health contract, a complete libsodium key-exchange/KDF/AEAD round trip, rejection of tampered ciphertext, and a production frontend build. Slice 2 adds: a migrated `users` table; Argon2id-hashed, rate-limited, uniqueness-enforced registration exercised end-to-end through Docker Compose against real Postgres; the crypto spike productionized into `keyExchange.ts` (role determination, a role-agnostic session-key wrapper, and base64 transport helpers) with all round-trip/tamper/replay tests preserved; `AuthScreen` wired to real registration with loading/error/success states and mismatched-password validation; and a trained, evaluated TF-IDF baseline.

## E2EE and client-side AI

The sender encrypts before network transmission. The server stores and relays only an authenticated encrypted envelope. The recipient verifies and decrypts locally, then runs scam classification on the recovered plaintext in ONNX Runtime Web before React renders it. This ordering preserves the server's inability to inspect content while enabling a non-blocking warning banner on the endpoint.

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

**Known limitation, stated plainly:** every training and evaluation row here comes from email and SMS corpora, not real chat messages. `data/label-schema.yaml`'s `evaluation_policy` already reserves the hand-curated chat-style set exclusively for out-of-domain evaluation and forbids training or tuning on it — that set does not exist yet (Slice 3 deliverable). Until it does, these numbers describe how well the model separates spam/phishing email and SMS from ham, not how it will perform on short, informal chat text, which differs sharply in register (fewer headers/boilerplate, shorter, more slang). Treat this baseline as a validated pipeline, not yet a validated chat-scam detector.

## Deployment

Deployment is scheduled for later slices after authentication, conversation-scoped storage, E2EE relay, model integration, and security verification are complete.

## Demo

A short two-browser E2EE and scam-warning demonstration will be recorded in the final slices.