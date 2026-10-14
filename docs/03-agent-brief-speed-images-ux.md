# Agent brief: speed, image hygiene, DistilBERT-only browser model, and chat UX

Implement the work in this file under `/home/sam/projects/Minor Project-II/` only. Do not commit unless the user asks. Do not start Slice 10 (hosted deploy). Do not change the README status line away from **Slice 9**, and do not rewrite the Deployment section into a hosted-deploy guide. This brief is post-Slice-9 product work. Update the README only after the code lands, and only to describe what actually shipped.

Read these before editing:

- `README.md` (architecture, threat model, status paragraph, pre-deployment hardening, ML section, Deployment)
- `frontend/src/hooks/useEncryptedConversation.ts`
- `frontend/src/hooks/useScamClassifierPreference.ts`
- `frontend/src/ml/scamClassifier.ts`
- `frontend/src/ml/scamBannerCache.ts`
- `frontend/src/ml/ortRuntime.ts`
- `frontend/src/chat/imageAttachment.ts`
- `frontend/src/api/conversationsClient.ts`
- `frontend/src/api/usersClient.ts`
- `frontend/src/components/ChatScreen.tsx`
- `frontend/src/components/SettingsPanel.tsx`
- `frontend/src/components/MessageActionsModal.tsx`
- `frontend/src/components/ChatMoreMenu.tsx`
- `frontend/src/components/Modal.tsx`
- `backend/app/routers/conversations.py`
- `backend/app/routers/users.py`
- `backend/app/services/relay.py` (`list_envelopes_page`, `hide_message_for_owner`)
- `backend/app/services/profiles.py`
- `backend/app/main.py` (CORS methods)
- `frontend/vite.precompressedOnnx.ts`
- `ml/reports/onnx_web_load_check.md`

Match the comment style already in the files you touch: this repo explains each line. Do not add `dangerouslySetInnerHTML`. Do not log plaintext, file keys, passwords, or scam scores. The server still stores and relays ciphertext only.

## Hard constraints

- End-to-end encryption stays intact. The server never decrypts a chat image or a message, never receives a scam score, and never learns a QR payload.
- Chat images use a random file key inside the encrypted envelope. File associated data is `['secure-chat-file-v1', conversationId, blobId]`. A blob ciphertext cannot be copied into another conversation. Forwarding means decrypt on this device, seal again for the destination conversation, upload a new blob, and send a new envelope.
- Avatars are public profile bytes (`users.avatar_bytes`), not chat envelopes. Magic-byte checks on avatars are allowed on the server. Chat-image checks run only after client decrypt.
- Do not add steghide, Autopsy, ClamAV, or a vision/malware model. Those tools do not fit this architecture. Steghide extracts hidden data only with a passphrase. Autopsy is a desktop disk-forensics app. A picture shown in an `<img>` tag is displayed, not executed.
- The browser loads one scam model: DistilBERT default (`distilbert_default`, 256-token int8, sidecar threshold 0.30, then ChatScreen requires P(scam) ≥ 0.35). It loads when the signed-in chat shell mounts. There is no model checkbox and no second graph.
- Do not load TF-IDF or the word BiLSTM in the browser at all. That includes ChatScreen, `ensureChatDefaultClassifier`, `loadTfidfCheckpoint`, `loadLstmCheckpoint`, `enableLstmOptIn`, and `/?mlLoadCheck=1`. Do not request `/ml/tfidf_best/`, `/ml/tfidf_default/`, `/ml/lstm_best/`, or `/ml/lstm_default/`. Do not create an ONNX session for those families. Remove their imports from the frontend bundle so Vite does not fetch their graphs.
- Python training reports under `ml/reports/` and `ml/scripts/` may stay on disk. They are not served to the chat app. Do not delete those offline reports in this task. Do not copy TF-IDF or LSTM ONNX files into `frontend/public/ml/` for the app to use.
- Do not switch the browser graph to `distilbert_best` (512 tokens, threshold 0.20). Do not keep `distilbert_best` as a second browser download.
- Do not weaken Argon2id. Leave epoch rotation at 50 messages or 24 hours.
- Banner cache (`localStorage`, per username, schema v2, max 500 rows) stores `{warned, revision, checkpointId}` only. A cache hit for the same id, revision, and checkpoint must not run the model again. Scroll-into-view is not a reason to rescore.
- CORS in `backend/app/main.py` already allows `GET`, `POST`, `DELETE`, and `PATCH`. Do not narrow that list. The README sentence that still says GET/POST only is stale; fix it when you update the README.

## Current behavior the new code must replace

Opening a chat in `useEncryptedConversation.openConversation` calls `fetchConversationMessages`, which GETs `/conversations/{id}/messages` with **no** `limit`. The handler already accepts `limit` (1–200) and `after` (oldest-first cursor) and returns `next_cursor`. That cursor walks **forward from the oldest** message. Using `after` for the first paint would show the start of the chat, not the latest messages. Newest-page loading needs a new `before` cursor (or an equivalent newest-first page) that still returns rows oldest-to-newest inside the page.

After decrypt, every uncached text row is classified immediately, and every image calls `hydrateImageAttachment`, which downloads the whole blob, decrypts it, and `setMessages`s the full list once per image.

`handleClearLocalTranscript` only clears React state and revokes `blob:` URLs. The status line says the server copy is unchanged. Reopening the chat loads every envelope again. Per-message hide already exists: `POST /conversations/{id}/messages/{message_id}/hide` inserts `message_hides (owner_id, message_id)`. Later history reads pass `exclude_hidden_for`. The peer is unaffected. There is no bulk hide.

Settings → Profile can replace an avatar via `POST /users/me/avatar` (JPEG/PNG/WebP, 200KB). There is no remove. The UI says “Change photo” and, when one exists, “A public photo is on this account.”

Image bubbles render `<img src={objectUrl}>` with no lightbox, zoom, or forward. `MessageActionsModal` refuses edit and copy when `attachment !== null`. Delete-for-everyone and Hide-for-me have no confirm step.

ChatScreen renders two checkboxes: “Use DistilBERT (large download)” and “Use Word BiLSTM Best”. They are mutually exclusive. TF-IDF loads on mount via `ensureChatDefaultClassifier`. DistilBERT runs unpadded in an ORT Web Worker (`ort.env.wasm.proxy = true`), with up to 4 WASM threads when `crossOriginIsolated` is true. `loadDistilbertCheckpoint` always fetches the graph. `enableDistilbertOptIn` is not idempotent. `vite.precompressedOnnx.ts` sends `model.onnx.br` (~37 MB) with `Content-Encoding: br` **only** for `npm run dev` and `npm run preview`. A static host that serves `model.onnx` as a normal file downloads the raw ~64 MB graph. The browser then holds ~64 MB either way; compression changes the transfer, not RAM.

Image messages are skipped by `needsClassify`. Nothing checks magic bytes. Nothing decodes a QR code. `qr.ae` in `urlFeatures.ts` is only a text-shortener domain.

## 1. Speed

Do these in order. Keep the encrypted send path correct while changing load behavior.

### 1.1 Newest page of history

- Extend `GET /conversations/{id}/messages` so the client can ask for the newest page. Suggested query: `limit` plus `before` (exclusive message id). Omit `before` to mean “newest `limit` rows”. Return those rows **oldest-first** plus `next_cursor` set to the oldest id in the page when older rows remain, otherwise null. Keep the existing `after` behavior for any caller that already uses it. Keep `exclude_hidden_for`.
- Default page size on the client: 50. Hard cap stays 200.
- `fetchConversationMessages` must send `limit`. On open, request the newest page only. Paint and decrypt that page, then open the WebSocket.
- Scrolling near the top loads the next older page (`before` = oldest id currently loaded), prepends it, and does not jump the viewport to the top.
- Delivered/read receipts on open cover the loaded page’s received ids, not an entire unpaged history.
- A live WebSocket envelope still appends. Do not refetch the whole transcript for one new message.
- First-open classification runs only on cache misses in the **loaded** page. Do not fetch older pages just to score them. When an older page arrives, score its cache misses once. A later reopen uses the banner cache and does not rescore hits.
- Add a backend test that the newest page is the latest envelopes, that `before` returns the next older page, that hidden rows stay hidden for that owner only, and that ciphertext-only responses are unchanged. Add a frontend test that open requests a limit and does not assume the full history.

### 1.2 Do not flood inference

- Keep the banner cache rules above. Trivial URL-free lines (`isTrivialShortNoUrl`) stay unscored and may still be cached as “do not warn”.
- When several cache misses are scored together (first open, or DistilBERT just finished loading), run them one at a time on the ORT worker. Paint each banner as it finishes. Do not start one `session.run` per message in the same tick.
- Score only after `distilbert_default` is loaded. While it is downloading, leave banners off. Do not fill that gap with TF-IDF or LSTM. Cache hits require `checkpointId === distilbert_default`. Older TF-IDF or LSTM cache rows are misses and are scored once with DistilBERT when that page is loaded. With pagination, that set is the loaded page, not the whole conversation.
- Overlap `fetchPublicKey` and `startOrFetchConversation` in `openConversation`. Fetch the epoch after the conversation id exists. History waits on the epoch and the session keys.

### 1.3 Lazy image bytes

- Do not GET every blob when the chat opens. Decrypt the envelope (it is a small JSON pointer) and leave `objectUrl` null until the bubble is near the viewport (`IntersectionObserver` or equivalent). Cap concurrent blob downloads at 3.
- Apply several finished URLs in one `setMessages` when they complete in the same turn, instead of one full-list copy per image.
- Revoke object URLs that scroll far away if many images are open, and hydrate them again if the user scrolls back. Failed decrypts stay on the existing failed placeholder and can be retried by the user.
- Leave Postgres `BYTEA` storage as it is. Do not move blobs to object storage in this task. A raw-byte download instead of base64 JSON is optional and only worth it if the image tests stay ciphertext-only; skip it if it complicates the API.

### 1.4 Leave these alone

- The ChatScreen 0.35 overlay on top of the DistilBERT sidecar threshold.
- DistilBERT int8, unpadded sequences, Web Worker, SIMD, and the thread cap.
- Argon2id `INTERACTIVE` / `MEMLIMIT_INTERACTIVE` on the vault.
- Epoch schedule.
- Server-side scam classification.

## 2. Browser model is DistilBERT only

### 2.1 Remove the other browser models

- Remove both ChatScreen checkboxes (“Use DistilBERT (large download)” and “Use Word BiLSTM Best”) and the preference hook’s TF-IDF / LSTM branches (`toggleLstm`, `toggleDistilbert`, `loadClassifierPreference`, `saveClassifierPreference`).
- On ChatScreen mount, load `distilbert_default` only. Show a non-blocking status such as “Loading scam detection on this device…” until that graph is ready, then “Scam detection is running on this device (not sent to the server).” A load failure leaves chat usable and banners off. It must not fall back to another model.
- Delete the stored classifier key (`secure-chat-classifier:<username>`) when chat loads so an old `tfidf` or `lstm` value cannot trigger a fetch.
- `MlLoadCheckScreen` (`/?mlLoadCheck=1`) must not load TF-IDF or LSTM. Either limit that page to `distilbert_default` or stop linking it from the chat UI. A browser network log of a chat session must not contain those checkpoint URLs.
- Stop importing `tfidf.ts` and `lstmTokenizer.ts` from the chat classifier path. Dead code that nothing imports is acceptable. Do not keep a dynamic `import()` that still fetches those graphs.
- Update `ChatScreen.test.tsx` so it expects no model checkboxes and expects classification to wait for DistilBERT. Update or remove `optInIds.test.ts` expectations that the chat bundle loads `tfidf_best` or `lstm_best`.

### 2.2 Make that single load cheaper

- If `distilbert_default` is already the resident session and the WordPiece vocab is cached in memory, return without fetching again. StrictMode’s double mount must share one in-flight load.
- After a successful fetch of `model.onnx` and the WordPiece sidecar, store those bytes in the Cache API (or an equivalent origin cache) keyed by `distilbert_default` plus a version from `manifest.json` (file name and byte length are enough if the manifest has no hash). The next visit on this browser reads the cache first and falls back to the network. Do not put plaintext messages in that cache. Do not set a year-long HTTP cache on the unhashed `model.onnx` name.
- Document in the README, in the new post-Slice-9 section: Vite adds `Content-Encoding: br` only for `npm run dev` and `npm run preview`. A production static host must do the same substitution itself (serve `model.onnx.br` for `Accept-Encoding: br` and set `Content-Encoding: br`) or the first DistilBERT download is the raw ~64 MB file. The Cache API avoids a second download in that browser. RAM after decode stays ~64 MB either way. Do not build that static host in this task. There is no TF-IDF graph beside it to unload.

## 3. Image hygiene (on device, after decrypt)

Run these in the browser after `openImageBytes` succeeds and before `createObjectURL`. Do not upload pixels, QR text, or scores.

### 3.1 Magic bytes

Accept the decrypted bytes only when the header matches the declared type:

| Declared type | Header |
| --- | --- |
| `image/jpeg` | `FF D8 FF` |
| `image/png` | `89 50 4E 47 0D 0A 1A 0A` |
| `image/gif` | `GIF87a` or `GIF89a` |
| `image/webp` | `RIFF` at offset 0 and `WEBP` at offset 8 |

Any other declared type, or a header that does not match, sets the bubble to the existing failed state and does not create an object URL. Unit-test the four accept cases and one mismatch.

### 3.2 QR code scored by the existing text model

- Decode QR codes from the decrypted image in the browser (a small library such as `jsQR` is fine). Do not fetch the URL.
- If a payload is a URL or other text, run it through DistilBERT once `distilbert_default` is loaded. If the graph is still downloading, wait for it and then score. Do not score the QR with TF-IDF or LSTM. A warn result uses the same non-blocking scam banner as text. The banner does not hide or delete the image.
- No QR means no extra classification. Image-pointer JSON stays unscored as chat text.
- Cache the banner under the message id, revision, and `distilbert_default`, so reopening does not decode the QR again when that cache hits. There is no model switch.

### 3.3 Re-encode still photos

- For JPEG, PNG, and WebP, draw the verified bitmap to a canvas and display the re-encoded blob (JPEG or PNG). That drops EXIF and most simple least-significant-bit hiding. The browser still decodes the original once; this is cleanup, not a sandbox.
- Do not re-encode GIF. Animation would collapse to one frame. GIF still gets the magic-byte check. QR decode may use one canvas frame.
- Put this helper next to `frontend/src/chat/imageAttachment.ts` and test the magic-byte gate without a real canvas when the test environment cannot draw.

## 4. Profile photo

- Add `DELETE /users/me/avatar`. It sets `avatar_bytes` and `avatar_media_type` to null for the signed-in user and returns the same profile shape as the other `/users/me` routes (`has_avatar: false`). Unknown users and missing tokens stay on the existing auth errors. Deleting when no avatar is stored is a successful no-op.
- Settings → Profile: keep “Change photo” (that is the edit/replace action). When `profileHasAvatar` is true, add “Remove photo”. Ask for confirmation with the existing `Modal`: the photo is public and removing it takes it off the account for everyone. On confirm, call DELETE, drop the local object URL, and show initials.
- `GET /users/{username}/avatar` already 404s when bytes are null. Avatar and contact list should refresh `has_avatar` after remove.
- Backend test: upload, GET with a bearer token, DELETE, GET returns 404, another user’s view follows `has_avatar`. Reject a non-image upload the way the current POST does. Do not store chat envelopes in this column.

## 5. Clear transcript for this user only

Replace the in-memory-only clear.

- Rename the menu label to something accurate, such as “Clear chat for me”.
- On click, open `Modal` before any request. The copy must say: this permanently removes the conversation from **your** account; the other person keeps their copy; you cannot undo it; messages that failed to decrypt are included. Cancel leaves the transcript untouched.
- On confirm, hide every message in this conversation for the **caller**. Prefer one authenticated endpoint, for example `POST /conversations/{id}/hide-all`, that inserts `message_hides` for each message id in that conversation for `current_user` and does not delete `messages` rows, blobs, or the conversation. Idempotent if some rows are already hidden. Do not broadcast a deletion frame. Do not include ciphertext in the response.
- Then clear the in-memory list and revoke object URLs, the way `handleClearLocalTranscript` does today.
- Reopening the chat must not bring those envelopes back (`exclude_hidden_for` already does this). The peer’s GET and live socket are unchanged.
- Pending local rows that have no server id are dropped locally only.
- Do not call delete-for-everyone. That removes the row for both people.
- Tests: caller’s history omits the hidden rows; peer’s history still returns them; cancel does not call the API; the menu no longer claims the server copy is unchanged.

## 6. Photos: expand, zoom, forward

Applies to sent and received images that have a successful `objectUrl`. Pending and failed bubbles are not openable.

- Clicking the image opens a dialog (lightbox) with the picture, a close control, and Escape / backdrop dismiss. Focus stays inside the dialog until it closes.
- Zoom: buttons for zoom in, zoom out, and reset, plus wheel or pinch if it stays simple. Clamp the scale (about 1× to 4×). Panning while zoomed can be a drag on the image. No new gallery framework.
- Forward: a contact picker limited to existing contacts. Confirm the destination. On confirm, decrypt is already done; seal a **new** blob with `sealImageBytes` for the **destination** `conversationId` and a new `blobId`, upload it, and send a new image envelope on that conversation (open it if needed, or send through the existing send-image path). The source blob row is not reused. The server never sees pixels or the file key. Show a short “Forwarded” failure message on the source chat if the upload fails, and leave the source bubble in place.
- Do not offer Forward for verification-failed rows.
- Add a confirm step on “Delete for everyone” in `MessageActionsModal`, using `Modal` or an inline confirm: this removes the message for both people. “Hide for me” can stay one click; it already matches per-user hide. Mention in the hide label that the other person still has it.

## 7. Further UX (same pass)

Ship these as well. They are small and match the chrome already in the app.

- **Scroll to latest.** When the user has scrolled up (including while an older page is loading), show a control that jumps to the newest bubble. Do not steal the scroll position when a new message arrives if the user is reading history; in that case the control can show that a new message arrived.
- **Retry image.** The failed placeholder offers “Retry” and calls hydrate again for that message id.
- **View avatar.** Clicking the large profile avatar (self in Settings, peer in contact info) opens the same lightbox without forward. Avatars are public; this does not decrypt a chat blob.
- **Date separators.** Between bubbles, show a short date label when the local calendar day changes. Use the existing `createdAt` string. Do not send that label to the server.

Do not add video, voice notes, stories, or a server-side image scanner.

## 8. README

After the code and tests pass, update `README.md`:

- Leave the status line starting with `Status: Slice 9.`
- In that status paragraph, state that the browser loads DistilBERT default only. TF-IDF and the word BiLSTM are not loaded in the chat app or on `/?mlLoadCheck=1`. Their offline training reports may remain in `ml/reports/`.
- Add a section **Post-Slice 9 speed, image hygiene, and UX** that records, briefly and factually: newest-page history, queued on-device DistilBERT scoring, lazy image download, DistilBERT as the only browser graph (Cache API and idempotent load), the static-host `Content-Encoding` limitation, magic-byte checks, QR text passed to DistilBERT, canvas re-encode for still photos, avatar remove, clear-chat-for-me via `message_hides`, and the image lightbox (zoom and forward-by-reseal).
- State again that image checks run after decrypt and that scores and QR text never leave the device.
- Fix the stale “CORS allows GET/POST only” wording so it matches `allow_methods` in `backend/app/main.py`.
- Leave the Deployment section as local Compose, with hosted Render/Railway/Fly.io still named as Slice 10.
- Do not claim a hosted `/health` check. Do not paste this brief into the README.

## 9. Verification

Run, and fix failures you cause:

```bash
cd backend && uv run pytest && uv run ruff check . && uv run mypy app tests
cd frontend && npm run test && npx tsc -b --noEmit
```

Also exercise the UI if a browser is available: open a chat (history request includes `limit`; the only model request is `distilbert_default`, with no `/ml/tfidf_*` or `/ml/lstm_*` request), reload and confirm the graph comes from the Cache API, scroll up for an older page, remove and replace an avatar, cancel then confirm clear-chat-for-me and confirm the other account still has the messages, open a photo, zoom it, and forward it to another contact. A forwarded image must decrypt in the destination chat and must fail if the destination associated data is wrong.

Ciphertext-boundary tests must still pass. Grep the server diff for plaintext fields, decryption calls, and score columns. There should be none on chat or blob routes.
