# Build Prompt: Secure Chat with AI-Based Phishing & Scam Detection

You are a senior full-stack engineer with deep security and applied-ML experience. Please help me Build a resume portfolio-grade, genuinely end-to-end encrypted chat application with an integrated AI-based phishing/scam message detector, following this specification exactly. Where something isn't fully specified, choose the option that is more secure, more standard, and more explainable in a README — never the option that's just fastest to implement. 

You will also be given the original prototype's `index.html`, `style.css`, and `script.js`.

> **How to use the old files — read carefully:** Use ONLY the visual design from these files — layout structure, color palette (purple gradient `#667eea → #764ba2`), corner radii, spacing, the lock-icon auth header, the floating-dot background animation, message bubble styling (gradient purple for sent, white/gray for received), the contacts sidebar, the dark-mode toggle, avatar circles with initials, the typing indicator, and the online-status dot. Rebuild these as clean, componentized React UI.
>
> Do **not** reuse any of the old JavaScript logic, the Firebase integration, the custom password hash function, the AES/CryptoJS encryption, the flat global-messages listener, or the localStorage-based contact storage. All of that is being replaced entirely by the security design in this document. If anything in the old `script.js` conflicts with a requirement below, this document wins.

---



## 1. Project summary

A real-time chat application where the server is cryptographically incapable of reading message content (true end-to-end encryption via client-side X25519 key exchange and authenticated encryption), combined with a trained ML classifier that flags likely phishing/scam messages — run entirely client-side, after decryption, so the AI feature never requires the server (or anyone) to see plaintext.

## 2. Security requirements

These are the project requirements. If there is any better suggestion or improvement, please suggest it first. Do not deviate from them even if a shortcut looks simpler without asking for permission:

- Password hashing **must** be Argon2id. Never MD5, SHA-256 alone, or any custom hash function.
- Authentication **must** use JWT access tokens (short-lived) + refresh tokens (longer-lived, rotated on use).
- Every user's X25519 keypair **must** be generated client-side. The private key **must never** be transmitted to, or stored on, the server — in any form, temporarily or otherwise.
- Session key derivation **must** use `crypto_kx` (or equivalent), not the raw ECDH shared secret used directly as a key.
- Message encryption **must** use an AEAD cipher — XChaCha20-Poly1305 (via libsodium). Never unauthenticated AES-CBC or any mode without a built-in integrity/authentication tag.
- The system **must** implement epoch-based key derivation for forward secrecy (fresh derived key per epoch, from the master session key, via `crypto_kdf_derive_from_key`).
- The server's database **must never** contain a table, column, or log line holding symmetric key material or plaintext message content. If you find yourself adding one "temporarily," stop and redesign instead.
- All message queries **must** be scoped by `conversation_id`. Never fetch the entire messages table to the client for local filtering.
- Login and registration endpoints **must** be rate-limited.
- No secrets, API keys, or credentials may be hardcoded anywhere in committed source. Use environment variables (`.env`, excluded via `.gitignore`, with a committed `.env.example`).
- The phishing/scam ML model **must** run client-side (ONNX Runtime Web), on already-decrypted plaintext, after the message reaches the recipient's device. The server must never receive plaintext message content for classification, or for any other purpose.



## 3. Tech stack

- **Frontend:** React (Vite), TypeScript preferred
- **Client-side crypto:** libsodium.js (`libsodium-wrappers`)
- **Client-side ML inference:** ONNX Runtime Web
- **Backend:** Python, FastAPI, native WebSocket support
- **Database:** PostgreSQL, SQLAlchemy (async), Alembic for migrations
- **Auth:** `argon2-cffi` for password hashing, `python-jose` for JWT
- **Rate limiting:** `slowapi`
- **ML training (offline, separate from the app runtime):** scikit-learn for the baseline model, PyTorch + HuggingFace Transformers for a fine-tuned DistilBERT model, `skl2onnx`/`torch.onnx` for export
- **Deployment:** Docker + Docker Compose (API + Postgres); Render/Railway/Fly.io for backend hosting, Vercel/Netlify for frontend
- **Version control:** Git/GitHub, with a portfolio-quality README



## 4. Architecture

```
Browser (React)                                    FastAPI server
  - libsodium: keypair, ECDH session keys   <-->      - Auth (Argon2id + JWT)
  - AEAD encrypt/decrypt (XChaCha20-Poly1305)          - Stores PUBLIC keys only
  - ONNX Runtime Web: local scam classifier <-->      - Stores ciphertext + nonce only
                                                         - Relays messages over WebSocket
                                                                    |
                                                              PostgreSQL
                                                       (users, conversations, messages —
                                                        no symmetric keys, ever)
```

The server's only responsibilities: authenticate users, store/serve public keys, store/relay ciphertext, coordinate the per-conversation epoch counter (a non-secret integer). It has no code path that touches plaintext or a decryption key.

## 5. Database schema

```sql
users
  id            uuid primary key
  username      text unique not null
  email         text unique not null
  password_hash text not null          -- Argon2id output, salt included
  public_key    text not null          -- base64 X25519 public key
  created_at    timestamptz not null default now()

conversations
  id            uuid primary key
  user_a_id     uuid references users(id)
  user_b_id     uuid references users(id)
  current_epoch int not null default 0
  created_at    timestamptz not null default now()

messages
  id              uuid primary key
  conversation_id uuid references conversations(id)
  sender_id       uuid references users(id)
  ciphertext      bytea not null
  nonce           bytea not null
  key_epoch       int not null
  created_at      timestamptz not null default now()

refresh_tokens
  id            uuid primary key
  user_id       uuid references users(id)
  token_hash    text not null
  expires_at    timestamptz not null
```



## 6. Cryptography implementation spec

1. **Keypair generation (client):** on registration or first login on a new device, generate an X25519 keypair via `sodium.crypto_kx_keypair()`. Upload only the public key (`POST /keys/me`). Persist the private key in IndexedDB, encrypted at rest with a key derived from the user's password (Argon2id, client-side); it must never be sent to the server.
2. **Public key lookup (client → server):** `GET /keys/{username}` — authenticated, returns the target user's public key. This is not secret data.
3. **Session key derivation (client only):** determine role by comparing usernames lexicographically (whoever sorts first = "client" for `crypto_kx` purposes — this only needs to be consistent, not meaningful). Call `crypto_kx_client_session_keys` / `crypto_kx_server_session_keys` accordingly to get directional session keys.
4. **Epoch key derivation (client only):** fetch the current epoch number from `GET /keys/conversations/{id}/epoch` (a plain integer, not secret), then derive `crypto_kdf_derive_from_key(32, epoch, "msgkey1", sessionKey)`.
5. **Message encryption (client):** `crypto_secretbox_easy(plaintext, randomNonce, epochKey)`. Send `{ciphertext, nonce, key_epoch}` over the WebSocket.
6. **Message decryption (client):** derive the same epoch key locally, call `crypto_secretbox_open_easy`; on authentication failure, surface a clear "message failed verification" state rather than showing corrupted output.
7. **Epoch rotation:** server increments `conversations.current_epoch` on a schedule (e.g., every 50 messages or every 24h) and broadcasts the bump over the WebSocket; it never generates or sees the key itself, only the counter.



## 7. AI/ML module spec

- **Training (offline, in a separate** `ml/` **project directory, not part of the runtime app):**
  - Baseline: TF-IDF + Logistic Regression or Linear SVM on combined SMS Spam Collection (UCI) + a public phishing/spam email dataset.
  - Upgrade: fine-tune DistilBERT on the same combined data.
  - Build a small hand-curated chat-style scam evaluation set (a few hundred examples) to sanity-check both models on text that actually resembles chat messages, since neither source dataset is native chat data.
  - Evaluate with precision, recall, F1, and a confusion matrix — not accuracy alone. Document the chosen classification threshold and the precision/recall trade-off reasoning in the README.
- **Export:** convert the trained model to ONNX (`skl2onnx` for the baseline, `torch.onnx.export` for DistilBERT).
- **Serving:** load the ONNX model once in the frontend via `onnxruntime-web`; run inference locally on each message's plaintext immediately after decryption, before render.
- **UX:** if the score exceeds the chosen threshold, render the message with a non-blocking "⚠️ This message shows signs of a scam" banner. Never hide, block, or auto-delete the message — the user stays in control.



## 8. UI/UX requirements

Rebuild the following as componentized React, matching the old prototype's visual language (see the note at the top of this document) but with clean component structure, no inline `onclick` strings, and proper accessibility (labels, `aria-` attributes where relevant):

- Auth screen: centered glass-morphic card, lock icon header, floating dot background, login/register toggle
- Chat screen: header with avatar/name/online-status/theme-toggle/logout, contacts sidebar, add-contact input, current-chat bar, typing indicator, message list with sent/received bubble styling, input area
- Dark mode toggle, persisted per user (not per browser via a single shared `localStorage` key that ignores the logged-in user — scope it correctly)
- Scam/phishing warning banner on flagged messages (new — not in the old prototype)



## 9. Project structure

```
/backend
  /app
    /models        # SQLAlchemy models
    /routers        # auth.py, keys.py, messages.py, ws.py
    /auth           # JWT + Argon2id logic
    /db.py
    /main.py
  /alembic
  /tests
  Dockerfile
  requirements.txt
/frontend
  /src
    /crypto         # keyExchange.js, and the ONNX inference wrapper
    /components      # AuthScreen, ChatScreen, ContactList, MessageBubble, etc.
    /api             # REST + WebSocket client wrappers
  vite.config.ts
/ml
  /notebooks         # training + evaluation notebooks
  /data
  /export            # scripts producing the .onnx files consumed by /frontend
docker-compose.yml
.env.example
README.md
```



## 10. Build order

Prerequisite: Suggest me different datasets that I should use to train the ML model for scam or phising detection and also suggest me more than 5 research papers regarding this topic to perform a literature review on past projects or research on similar topic as is this. 

Work in the following phases carefully and simultaneously. **After finishing atleast 10% of each phase, stop, summarize what you built, how to run/test it, and wait for confirmation before continuing** — do not build a complete phase in one pass.

1. **Backend foundations:** FastAPI project skeleton, Postgres models, Argon2id register/login, JWT issuing/verification, `slowapi` rate limiting. No crypto or ML yet.
2. **E2EE core:** key upload/lookup endpoints, WebSocket ciphertext relay, epoch endpoint — plus the client-side `crypto/keyExchange.js` module (keypair gen, session key derivation, epoch key derivation, encrypt/decrypt). Prove two browser sessions can hold an encrypted conversation through the server, using a minimal throwaway UI.
3. **Frontend proper:** rebuild the UI per §8, wired to the real backend from phases 1–2.
4. **ML track:** dataset prep, baseline model + evaluation notebook, DistilBERT fine-tune, ONNX export, `onnxruntime-web` integration, warning-banner UX.
5. **Hardening & polish:** epoch rotation scheduling, input validation everywhere, Docker Compose, deployment, README (architecture diagram, threat model, explicitly documented scoping decisions), short demo recording.



## 11. Explicit anti-patterns — do not reintroduce these

- Any database rule/config that allows unauthenticated or overly broad read/write access
- Any symmetric key or plaintext message stored or logged server-side, even temporarily (e.g., in error logs, debug prints, or exception messages)
- Rendering any user-controlled string via `dangerouslySetInnerHTML` or raw `innerHTML`
- A single flat "all messages" query/listener that relies on client-side filtering
- Password policies weaker than a reasonable minimum length with no complexity theater (length + Argon2id matter far more than forced special characters)
- Comparing secrets/hashes with plain `===` where a timing-safe comparison is warranted (the auth libraries above already handle this — don't hand-roll it)



## 12. Deliverables checklist

- [ ] `docker-compose up` brings up API + Postgres locally with no manual steps beyond copying `.env.example` to `.env`
- [ ] README with: architecture diagram, explicit threat model (what this protects against and what it deliberately doesn't), setup instructions, and a clearly written explanation of how the E2EE-vs-AI-scanning conflict was resolved
- [ ] `.env.example` covering every required environment variable, no real secrets committed
- [ ] Basic automated tests: auth flow, and a crypto round-trip test (encrypt → decrypt → assert original plaintext, plus a tampered-ciphertext test that asserts decryption fails)
- [ ] ML evaluation notebook/report checked into `/ml`, with precision/recall/F1 and the threshold justification