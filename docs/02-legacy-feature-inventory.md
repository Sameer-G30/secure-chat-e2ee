# Legacy frontend feature inventory and Firebase-to-current mapping

This document is the Phase 1 deliverable required before any legacy-frontend porting
work began: a full inventory of what `Legacy files/` actually implements, the Firebase
mechanism each feature depends on, and the equivalent mechanism in the current
(non-Firebase) FastAPI + Postgres + WebSocket + client-side-crypto stack. Scope for the
port itself (confirmed separately) is **everything except email verification and
password reset** — those need an SMTP provider and are out of scope for this project.

`Legacy files/` actually contains **two** separate prototypes against the same Firebase
project (`encrypted-chat-56ab1`): a vanilla HTML/CSS/JS prototype at the folder root, and
a more complete React + Vite rewrite under `Legacy files/frontend/`. Only Firebase
Realtime Database is used by either — no Firestore, Storage, Cloud Functions, or FCM.

## 1. Feature inventory

| # | Feature | Legacy implementation | Present today? |
| --- | --- | --- | --- |
| 1 | Register / log in / log out | Both apps | Yes (Argon2id + JWT) |
| 2 | Server-side contacts | React only (vanilla used localStorage) | Yes |
| 3 | Realtime 1:1 messaging | Both apps | Yes (E2EE + WebSocket relay) |
| 4 | Typing indicator | Both apps | Yes (metadata only) |
| 5 | Online/offline presence | Both apps | Yes (metadata only) |
| 6 | Dark mode | Both apps | Yes (per-username) |
| 7 | User search by username | React (full-table scan) | **No** — exact-match only |
| 8 | Contact delete | Neither (vanilla: localStorage remove only) | **No** — add-only |
| 9 | Message edit | React (buggy — see below) | **No** |
| 10 | Message delete for me | React (buggy — see below) | **No** |
| 11 | Message delete for everyone | React | **No** |
| 12 | Clear entire chat | React | **No** |
| 13 | Export chat to `.txt` | React | **No** |
| 14 | In-chat message search | React (filter only, no scroll-to-result) | **No** |
| 15 | Block / unblock user | React (**localStorage only**, not server-enforced) | **No** |
| 16 | Report user | React (metadata to `reports/`) | **No** |
| 17 | Settings panel (theme/general/account stubs) | React | **No** |
| 18 | Password strength meter | React | **No** |
| 19 | Initials avatars | Both apps | Partially (already in current ChatScreen) |
| 20 | Email verification | React (Firebase Auth) | **Excluded from this port** (needs SMTP) |
| 21 | Password reset | React (Firebase Auth) | **Excluded from this port** (needs SMTP) |
| 22 | File/image attachments | React (attach button, **no handler** — never worked) | **Excluded** (never functional in the legacy app either) |
| 23 | Group chats | Neither | Out of scope for both |

Rows 7-19 are the Phase 1 delta this port implements. Rows 20-22 are explicitly flagged
as not cleanly replicable in this slice.

## 2. Old mechanism -> new mechanism mapping

| Feature | Legacy Firebase mechanism | Current-stack equivalent |
| --- | --- | --- |
| Auth | React: Firebase Auth (`createUserWithEmailAndPassword`, `signInWithEmailAndPassword`, `onAuthStateChanged`). Vanilla: custom RTDB row + a bit-shift "hash." | Already ported: Argon2id password hashing, PyJWT access/refresh with rotation-on-use, `POST /auth/register` / `/auth/login`. |
| Identity keys | React: X25519 keypair generated client-side; public key to `users/{uid}/publicKey`; **private key in plaintext localStorage**. | Already ported and materially stronger: private key sealed with Argon2id-derived XChaCha20-Poly1305 in IndexedDB (`crypto/keyVault.ts`), never in `localStorage`. |
| User search | `users` RTDB node `.get()` — downloads **every** user row client-side, substring-matches in JS. | New `GET /users/search?q=&limit=` — server-side prefix match via a `lower(username)` index, authenticated, rate-limited, capped result count. Strictly narrower than the legacy behavior. |
| Contacts (add/list) | React: `contacts/{ownerUid}/{contactUid}` RTDB node, `onChildAdded` listener. Vanilla: `localStorage`. | Already ported: `contacts` Postgres table, `GET/POST /contacts`. |
| Contact delete | Not implemented in either legacy app for the server-backed (React) version. | New `DELETE /contacts/{username}`. |
| Realtime messages | React: `messages/{conversationId}/{pushId}` RTDB node, `onChildAdded` + `limitToLast(50)`. Vanilla: a single flat `messages/` node, client-filtered by sender/receiver (leaks metadata to every connected client). | Already ported and strictly better: `messages` Postgres table scoped by `conversation_id`, authenticated `WS /ws/conversations/{id}` relay, server never sees plaintext either way (legacy AES key was server-visible; current XChaCha20-Poly1305 session keys never leave the browser). |
| Typing / presence | RTDB `typing/{a}_{b}` node / `users/{uid}/online`. | Already ported: WebSocket metadata frames (`type: "typing"` / `"presence"`), never persisted. |
| Message edit | RTDB `.update({ ciphertext, nonce, edited, editedAt, epoch })`. **Broken in the legacy app**: UI reads `isEdited`, the write sets `edited` — the indicator never actually showed. | New `PATCH`-equivalent flow requires an associated-data fix first — see §3 below. Ships only if that fix is approved. |
| Delete for me | RTDB `.update({ deleted: true, deletedBy })`. **Broken in the legacy app**: UI reads `isDeleted`, the write sets `deleted`. | New `message_hides` table (owner-scoped), filtered out of that owner's history query. Fixes the field-name bug rather than porting it. |
| Delete for everyone | RTDB `.remove()`. | New `DELETE /conversations/{id}/messages/{message_id}`, sender-only, hard row delete, broadcast `{type: "message_deleted", id}` to the peer. |
| Clear chat | RTDB `messages/{conversationId}` `.remove()`. | Client-side only: clears this tab's in-memory transcript. A true "delete my copy of this whole conversation" needs the same `message_hides` mechanism as delete-for-me, applied to every row; deferred as a fast-follow rather than blocking this phase. |
| Export chat | Client-side `Blob` download of decrypted text already in memory. | Same mechanism — this never touched Firebase and ports unchanged, behind an explicit "this writes plaintext to disk" warning that the legacy app did not show. |
| In-chat search | Client-side filter of the in-memory `chatMessages` array. | Same mechanism — client-side filter over already-decrypted state; no server involvement either version. |
| Block / unblock | **`localStorage` only** — never synced, never enforced server-side; a blocked user could still send via RTDB. | New `blocks` Postgres table, enforced at conversation creation, contact add, and relay (silent drop, sender still sees `accepted` so the block is not disclosed). Strictly stronger than the legacy behavior, which enforced nothing. |
| Report user | RTDB `reports/{pushId}` (`reporterUid`, `reportedUid`, `reason`, `timestamp`, `status`). | New `reports` Postgres table, metadata-only by design (see §3). |
| Settings panel | Local component state; `Profile` / `Privacy` / `Account` / `Storage` / `Help` are `alert('coming soon')` stubs; only theme and logout actually work. | New settings panel with the same shell; only theme (including the legacy's **system** option, not currently in `theme.ts`), and logout are wired to real behavior — the other panels stay explicit "not implemented" rather than a silent dead button. |
| Password strength meter | Client-only heuristic in `useAuthForms.js`. | Same mechanism — no backend involvement in either version; ports directly into `AuthScreen.tsx`. |

## 3. Features flagged as not cleanly replicable (and why)

- **Email verification / password reset.** Firebase Auth provided both for free. Neither
  has an equivalent without adding an SMTP-sending service (SendGrid, SES, or similar)
  and its associated secret management, which is out of scope per the agreed Phase 1
  boundary. Registration keeps working exactly as today (no verification gate).
- **File/image attachments.** The legacy attach button (`ChatInput.jsx`) had no `onClick`
  handler in either app — it never worked, so there is nothing functional to port. A real
  implementation would also need to decide how encrypted file storage interacts with the
  E2EE model (Firebase Storage was configured but unused), which is a separate design
  question outside this phase.
- **Report content attachment.** The legacy report only ever carried metadata
  (`reporterUid`, `reportedUid`, `reason`) — it could not attach the reported message text
  either, because Firebase Storage/Firestore were never wired for it. This project's E2EE
  guarantee makes that limitation a hard requirement, not just an omission: the server has
  no key to read message ciphertext, so it structurally cannot attach message evidence to
  a report without breaking the trust boundary in the architecture diagram. Reports here
  are metadata-only by design, with UI copy that says so explicitly, matching (and slightly
  improving the honesty of) the legacy behavior.

## 4. The one crypto-affecting decision: message editing

The AEAD associated data built in `frontend/src/crypto/keyExchange.ts`
(`encodeAssociatedData`, lines ~185-194) is:

```
['secure-chat-envelope-v1', conversationId, senderId, keyEpoch]
```

It binds no message identity and no revision number. Two consequences, one already true
today and one introduced by porting message editing at all:

- **Already true today (pre-existing, not introduced by this phase):** within one
  conversation/sender/epoch, a malicious or compromised server can reorder, duplicate, or
  drop envelopes and every one still verifies, because nothing in the associated data
  distinguishes one envelope from another with the same `(conversation_id, sender_id,
  key_epoch)`.
- **Introduced by editing, if shipped naively:** the server could serve the *pre-edit*
  ciphertext after a real edit and the client would accept it as authentic — a silent
  edit-rollback oracle, because "this is the current version of this message" is not
  something the AEAD tag protects.

**Decision needed before implementing editing:** extend the associated data to a new
`v2` format — `['secure-chat-envelope-v2', conversationId, senderId, keyEpoch, messageId,
revision]`, with the client generating `messageId` (a UUID) at send time. This is a
breaking crypto-format change requiring an `ad_version` column so `v1` history still
decrypts and all new sends use `v2`. This closes both issues above, not just the one
editing introduces.

If the AD change is deferred, delete-for-everyone still ships (it does not need message
identity in the AD — a hard row delete plus a broadcast is enough), but message *editing*
does not, and the rollback-oracle risk is documented as a known limitation rather than
silently shipped. This decision is tracked and resolved in the next phase step.

**Decision taken: implement the `v2` associated-data extension.** The confirmed Phase 1
scope explicitly includes message edit, and the `v2` fix closes both the edit-specific
rollback oracle and the pre-existing reorder/duplicate/drop gap in one change, rather than
shipping editing with a known oracle. Implementation: `messages.ad_version` (`smallint`,
default `1`) plus `messages.client_message_id` (`uuid`, nullable — only populated for
`v2` rows) columns; `RelayEnvelopeIn` accepts an optional `message_id` field the client
generates; the relay stores it and echoes it back so all peers derive the same AD;
existing `v1` history keeps decrypting exactly as before (its AD never included a message
id, and it is never rewritten). See `backend/app/schemas/messages.py`,
`backend/app/models/message.py`, and `frontend/src/crypto/keyExchange.ts` for the
implementation, and `backend/alembic/versions/` for the migration.

## 5. Legacy bugs fixed rather than ported

- `deleted` (write) vs `isDeleted` (read) and `edited` (write) vs `isEdited` (read) field
  mismatches — the legacy UI never actually showed either indicator.
- Unvalidated confirm-password field on registration.
- `getEpochKey('global', ...)` hardcoded instead of using the actual conversation id.
- `clearEpochKeys` imported but never exported — a runtime error on `clearChat()`.
- Encryption call-site argument-order mismatch between `messaging.js` and `encryption.js`.
- Missing `libsodium-wrappers` dependency (imported, never added to `package.json`) — the
  legacy React app could not actually build/run its E2EE path as committed.

None of these are ported; each is either fixed in the new implementation or made moot by
reusing the current (already-correct) crypto and contacts modules instead of the legacy
ones.
