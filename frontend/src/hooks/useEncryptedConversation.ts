// Extracted from ChatScreen.tsx during the pre-deployment refactor (Phase 3a), then
// extended during Phase 1 (legacy feature port) with safe message editing and
// delete-for-everyone/delete-for-me. This is the core of the conversation state
// machine: look up a peer, derive session keys, load ciphertext-only history, open the
// relay WebSocket, encrypt/send/edit/delete, and decrypt/verify/classify everything
// that arrives.

// Import React's state/effect/ref hooks used to drive the live conversation.
import { useEffect, useRef, useState } from 'react'
// Import the form-event type separately; verbatimModuleSyntax requires type-only imports.
import type { FormEvent } from 'react'

// Import the public-key lookup used before session-key derivation.
import { fetchPublicKey, KeysApiError } from '../api/keysClient'
// Import conversation create/fetch, the spec §6.4 epoch reader, and the Phase 1
// delete-for-everyone / hide-for-me REST calls.
import {
  ConversationsApiError,
  deleteConversationMessage,
  fetchConversationEpoch,
  fetchConversationMessages,
  hideConversationMessage,
  startOrFetchConversation,
} from '../api/conversationsClient'
// Import the authenticated ciphertext-only WebSocket wrapper.
import { connectChatSocket } from '../api/chatSocket'
import type { ChatSocket, RelayedEnvelope, RelayedMessageDeleted } from '../api/chatSocket'
// Import the shared authorized-request contract this hook is given by the caller.
import type { AuthorizedRequestApi } from './useAuthorizedRequest'
// Import the production crypto helpers used to encrypt before send and verify on receive.
import {
  decodeBase64,
  decryptMessage,
  deriveSessionKeys,
  encryptMessage,
  initializeSodium,
} from '../crypto/keyExchange'
import type { DirectionalSessionKeys } from '../crypto/keyExchange'
import type { CheckpointId, ClassifyResult } from '../ml/types'
import { CHATSCREEN_DEFAULT_ID } from '../ml/types'
import {
  readCachedBanner,
  renameCachedBanner,
  writeCachedBanner,
} from '../ml/scamBannerCache'
// Reuse AuthContext's own Session shape instead of declaring a second, structurally
// coincidental copy of the same four fields.
import type { Session } from '../context/AuthContext'

// Describe one bubble in the in-memory message list (no server-side pagination yet).
export interface ChatMessage {
  // Identify the row for React reconciliation. A freshly sent message uses a
  // temporary local id until the server's "accepted" ack supplies the real row id
  // (see pendingSendQueueRef below) — delete/edit REST calls need that real id.
  id: string
  // Distinguish sent (purple) from received (gray) bubbles using the legacy palette.
  direction: 'sent' | 'received'
  // Carry verified plaintext, or null when authentication failed.
  plaintext: string | null
  // Signal that decrypt+verify failed; the UI must not invent corrupted text.
  verificationFailed: boolean
  // Non-blocking banner after local classification of verified plaintext.
  scamWarning: boolean
  // Carry the client-chosen v2 message identity (bound into the AD), or null for a
  // v1 envelope. Required to send an edit of this message.
  clientMessageId: string | null
  // Count edits to this message; 0 for the original send/receive.
  revision: number
  // Carry the most recent edit time, or null if never edited.
  editedAt: string | null
  // True only for an own just-sent message still waiting on the server's "accepted"
  // ack (and therefore not yet safe to edit/delete, since its real row id is unknown).
  pending: boolean
}

// Describe the live conversation this tab has derived keys for.
export interface ActiveChat {
  // Bind AEAD associated data and WebSocket routing to this conversation UUID.
  conversationId: string
  // Display the peer handle in the header.
  peerUsername: string
  // Bind AEAD associated data to this tab's user UUID.
  selfId: string
  // Identify the peer UUID for received-envelope associated data.
  peerId: string
  // Pass the server's current epoch into the NEXT encrypt only; decrypt uses envelope.keyEpoch.
  currentEpoch: number
  // Hold directional crypto_kx keys derived locally; never sent to the server.
  sessionKeys: DirectionalSessionKeys
}

// Describe what this hook hands back to ChatScreen.
export interface EncryptedConversationApi {
  activeChat: ActiveChat | null
  messages: ChatMessage[]
  draft: string
  isConnecting: boolean
  peerOnline: boolean
  peerTyping: boolean
  openConversation: (peerUsername: string) => Promise<void>
  handleSend: (event: FormEvent<HTMLFormElement>) => void
  handleDraftChange: (value: string) => void
  // Re-encrypt and resend an already-accepted own message with the same v2 message
  // identity and an advanced revision. No-op if the message is pending, not sent by
  // this account, verification-failed, or was never sent as v2 (clientMessageId null
  // — a message this old client sent before editing existed cannot be edited).
  handleEditMessage: (id: string, newPlaintext: string) => Promise<void>
  // Hard-delete an own message for every participant. No-op under the same guards as
  // edit (except revision/v2, since delete does not need message identity in the AD).
  handleDeleteMessage: (id: string) => Promise<void>
  // Hide a message (own or the peer's) from this account's own future history only.
  handleHideMessage: (id: string) => Promise<void>
  // Drop this tab's in-memory transcript without touching the server (legacy "clear
  // chat" was a Firebase wipe; here it is local-only and documented as such).
  handleClearLocalTranscript: () => void
  // Tear down the live socket and forget this conversation locally (used after a
  // server-side block, or after removing the currently open contact).
  closeConversation: () => void
}

// The server's WebSocket auth-failure close code (see backend/app/routers/ws.py).
const WS_CLOSE_UNAUTHORIZED = 4401

// Drive one ChatScreen's live encrypted conversation: lookup, key derivation, history,
// relay socket, send/edit/delete/receive, epoch rotation, and local classification.
//
// `session` is nullable (and this hook must still be called unconditionally, on every
// render, per the Rules of Hooks) purely to survive the one render where AuthContext's
// session flips back to null on logout, right before AppRoutes swaps ChatScreen back
// out for AuthScreen. ChatScreen itself still only calls openConversation/handleSend
// while a session exists, so the null-session branches below are a safety net, not a
// real code path.
export function useEncryptedConversation(
  session: Session | null,
  authorizedRequest: AuthorizedRequestApi<Session>,
  classify: (plaintext: string) => Promise<ClassifyResult | null>,
  setStatusMessage: (message: string) => void,
  setErrorMessage: (message: string | null) => void,
  // Bump when the ready scam model changes so cache misses are re-scored (banners can clear).
  classifierGeneration: number = 0,
  // Catalog id for cache hits (DistilBERT default while the graph is still loading).
  scoringCheckpointId: CheckpointId = CHATSCREEN_DEFAULT_ID,
): EncryptedConversationApi {
  const [isConnecting, setIsConnecting] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState('')
  const [activeChat, setActiveChat] = useState<ActiveChat | null>(null)
  const [peerOnline, setPeerOnline] = useState(false)
  const [peerTyping, setPeerTyping] = useState(false)

  // Keep the live socket off React state so reconnects do not re-render on every frame.
  const socketRef = useRef<ChatSocket | null>(null)
  // Keep the latest active chat in a ref so socket callbacks never close over a stale epoch.
  const activeChatRef = useRef<ActiveChat | null>(null)
  // Mirror `messages` into a ref so handleIncomingEnvelope's dedup/edit check can read
  // the current transcript synchronously. A `setMessages(updater)` functional update's
  // updater callback is not guaranteed to run before this function returns, so control
  // flow (deciding whether to classify) must never depend on a value set inside it.
  const messagesRef = useRef<ChatMessage[]>([])
  // Debounce the typing-false frame so keystrokes do not flood the metadata channel.
  const typingTimeoutRef = useRef<number | null>(null)
  // FIFO queue of this tab's own sent (not edited) message local ids awaiting the
  // server's "accepted" ack. One WebSocket connection processes frames in the order
  // sent and acks them in that same order, so the oldest queued local id always
  // corresponds to the next "accepted" frame that carries a real, not-yet-seen row id.
  const pendingSendQueueRef = useRef<string[]>([])
  // Map optimistic send ids to server ids so a slow DistilBERT result still finds the row.
  const acceptedIdsRef = useRef<Map<string, string>>(new Map())
  // Always call the latest classify() from socket/history callbacks (the function identity changes).
  const classifyRef = useRef(classify)
  // Drop stale classify results after the ready model changes (DistilBERT load vs TF-IDF).
  const classifySeqRef = useRef(0)
  // Latest handle for banner-cache keys (socket callbacks must not close over a stale session).
  const usernameRef = useRef(session?.username)
  // Latest catalog id so hydrate can paint DistilBERT banners before WASM is ready.
  const scoringCheckpointRef = useRef(scoringCheckpointId)

  // Keep classifyRef current every render so hydrate/socket paths do not close over TF-IDF forever.
  classifyRef.current = classify
  // Keep the cache username in lockstep with the signed-in handle.
  usernameRef.current = session?.username
  // Keep the cache checkpoint in lockstep with the checkbox (not the WASM-ready flag).
  scoringCheckpointRef.current = scoringCheckpointId

  // Mirror active chat into the ref whenever React state changes.
  useEffect(() => {
    activeChatRef.current = activeChat
  }, [activeChat])

  // Mirror messages into the ref whenever React state changes (see messagesRef's doc comment).
  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  // Close any open relay socket when this screen unmounts (logout or session end).
  useEffect(() => {
    return () => {
      socketRef.current?.close()
      socketRef.current = null
      if (typingTimeoutRef.current !== null) {
        window.clearTimeout(typingTimeoutRef.current)
      }
    }
  }, [])

  // Look up a prior banner for this id+revision+checkpoint; null means classify.
  function cachedWarning(messageId: string, revision: number): boolean | null {
    // No signed-in handle means we cannot read a per-user cache key.
    const username = usernameRef.current
    // Skip storage when ChatScreen is in the logout render.
    if (!username) {
      // Treat as a miss so a later classify still runs.
      return null
    }
    // Match checkpoint so DistilBERT default cannot reuse a TF-IDF row.
    return readCachedBanner(username, messageId, revision, scoringCheckpointRef.current)
  }

  // Paint a cached DistilBERT/TF-IDF/LSTM banner without waiting for WASM.
  function applyCachedBanner(row: ChatMessage): ChatMessage {
    // Verification failures never show a banner and are never cached.
    if (row.verificationFailed || row.plaintext === null) {
      // Leave the row unchanged.
      return row
    }
    // Read last session's decision for this catalog id.
    const warned = cachedWarning(row.id, row.revision)
    // Cache miss: keep whatever the caller set (usually false) until classify returns.
    if (warned === null) {
      // New or edited plaintext still needs the ready model.
      return row
    }
    // Reload path: show the banner immediately.
    return { ...row, scamWarning: warned }
  }

  // True when this bubble has no matching cache row for the current checkpoint.
  function needsClassify(row: ChatMessage): boolean {
    // Skip verification failures; they must never reach the classifier.
    if (row.verificationFailed || !row.plaintext) {
      // No plaintext to score.
      return false
    }
    // Cache hit: skip WASM on reload (DistilBERT session create is ~1s plus infer/msg).
    return cachedWarning(row.id, row.revision) === null
  }

  // Classify verified plaintext locally. Banners may turn on *or* off when the model changes.
  function classifyAndFlag(messageId: string, plaintext: string, revision: number) {
    // Capture the model generation so a DistilBERT result cannot overwrite a later TF-IDF pass.
    const seq = classifySeqRef.current
    // Run after decrypt/send so the bubble appears before WASM returns.
    void classifyRef.current(plaintext)
      .then((result) => {
        // Ignore results from a previous DistilBERT/TF-IDF/LSTM generation.
        if (seq !== classifySeqRef.current) {
          // A newer re-score is in flight or already applied.
          return
        }
        // null means the requested heavy graph is still loading (or classify failed): leave the flag.
        if (result === null) {
          // Cache misses are classified when classifierGeneration bumps after load.
          return
        }
        // Remember this decision so the next reload does not wait on WASM.
        const username = usernameRef.current
        // Prefer the server id if the accepted ack already renamed this send.
        const resolvedId = acceptedIdsRef.current.get(messageId) ?? messageId
        // Only persist when a handle exists (not the logout render).
        if (username) {
          // Store warned/revision/checkpointId; never plaintext.
          writeCachedBanner(username, resolvedId, revision, result.checkpointId, result.warned)
        }
        // Apply the ready model's decision, including clearing a TF-IDF false alarm.
        setMessages((existing) =>
          existing.map((message) =>
            message.id === resolvedId && message.scamWarning !== result.warned
              ? { ...message, scamWarning: result.warned }
              : message,
          ),
        )
      })
      .catch(() => {
        // Classification is non-blocking; a thrown WASM error must not surface as an unhandled rejection.
      })
  }

  // Re-score cache misses when DistilBERT/LSTM becomes ready or is turned off.
  useEffect(() => {
    // Invalidate in-flight scores from the previous model.
    classifySeqRef.current += 1
    // Apply cached banners for the new checkpoint (reload / toggle) without WASM.
    setMessages((existing) => existing.map((row) => applyCachedBanner(row)))
    // Walk the live transcript (may be empty before a conversation is opened).
    for (const row of messagesRef.current) {
      // Skip rows whose last DistilBERT/TF-IDF decision is already on disk.
      if (needsClassify(row) && row.plaintext) {
        // Score with the model that is ready now.
        classifyAndFlag(row.id, row.plaintext, row.revision)
      }
    }
    // classifierGeneration / scoringCheckpointId are the triggers; classifyAndFlag reads refs.
  }, [classifierGeneration, scoringCheckpointId])

  // Decrypt one envelope with the matching directional key, or record a verification failure.
  function decryptEnvelope(envelope: RelayedEnvelope, current: ActiveChat): ChatMessage {
    const direction: 'sent' | 'received' =
      envelope.senderId === current.selfId ? 'sent' : 'received'
    const key =
      envelope.senderId === current.selfId
        ? current.sessionKeys.transmitKey
        : envelope.senderId === current.peerId
          ? current.sessionKeys.receiveKey
          : null
    // v2 envelopes must reconstruct the exact AD used at encryption time, including
    // the message identity and revision; v1 envelopes (messageId null) omit both.
    const identity =
      envelope.messageId !== null
        ? { messageId: envelope.messageId, revision: envelope.revision }
        : {}
    if (key === null) {
      return {
        id: envelope.id,
        direction: 'received',
        plaintext: null,
        verificationFailed: true,
        scamWarning: false,
        clientMessageId: envelope.messageId,
        revision: envelope.revision,
        editedAt: envelope.editedAt,
        pending: false,
      }
    }
    try {
      const plaintext = decryptMessage(
        {
          ciphertext: envelope.ciphertext,
          nonce: envelope.nonce,
          keyEpoch: envelope.keyEpoch,
        },
        key,
        {
          conversationId: current.conversationId,
          senderId: envelope.senderId,
          ...identity,
        },
      )
      return applyCachedBanner({
        id: envelope.id,
        direction,
        plaintext,
        verificationFailed: false,
        scamWarning: false,
        clientMessageId: envelope.messageId,
        revision: envelope.revision,
        editedAt: envelope.editedAt,
        pending: false,
      })
    } catch {
      return {
        id: envelope.id,
        direction,
        plaintext: null,
        verificationFailed: true,
        scamWarning: false,
        clientMessageId: envelope.messageId,
        revision: envelope.revision,
        editedAt: envelope.editedAt,
        pending: false,
      }
    }
  }

  // Decrypt a live envelope. Upserts by id so an edit (same id, higher revision)
  // replaces the existing row in place instead of appending a duplicate, while a
  // stale/out-of-order redelivery (same or lower revision) is ignored.
  function handleIncomingEnvelope(envelope: RelayedEnvelope) {
    const current = activeChatRef.current
    if (current === null) {
      return
    }
    if (envelope.senderId === current.selfId) {
      // The server never echoes the sender's own envelope back; guard anyway.
      return
    }
    // Decide skip/append/replace from the ref (always current), not from inside the
    // setMessages updater below, whose execution timing relative to this function
    // returning is not guaranteed (see messagesRef's doc comment).
    const existingIndex = messagesRef.current.findIndex((message) => message.id === envelope.id)
    if (existingIndex !== -1 && messagesRef.current[existingIndex].revision >= envelope.revision) {
      // A duplicate delivery, or an edit arriving out of order; keep what we have.
      return
    }
    const row = decryptEnvelope(envelope, current)
    setMessages((existing) => {
      const index = existing.findIndex((message) => message.id === envelope.id)
      if (index === -1) {
        return [...existing, row]
      }
      const updated = [...existing]
      updated[index] = row
      return updated
    })
    if (needsClassify(row) && row.plaintext) {
      classifyAndFlag(row.id, row.plaintext, row.revision)
    }
  }

  // Drop a hard-deleted message from the transcript when the peer (or another of
  // this account's own tabs) deletes it; the deleting tab already removed its own
  // copy optimistically in handleDeleteMessage.
  function handleMessageDeleted(deletion: RelayedMessageDeleted) {
    if (activeChatRef.current?.conversationId !== deletion.conversationId) {
      return
    }
    setMessages((existing) => existing.filter((message) => message.id !== deletion.id))
  }

  // Update the in-memory encrypt epoch after a WS bump or a GET .../epoch refetch.
  function applyCurrentEpoch(nextEpoch: number) {
    // Read the live conversation so socket callbacks never close over a stale render.
    const current = activeChatRef.current
    // Ignore a bump that arrives before session keys exist.
    if (current === null) {
      return
    }
    // Ignore a stale counter that would encrypt with an older subkey id.
    if (nextEpoch < current.currentEpoch) {
      return
    }
    // Keep directional session keys; only the KDF subkey id changes.
    const updated: ActiveChat = { ...current, currentEpoch: nextEpoch }
    // Encrypt must use the new id on the next send, even before React re-renders.
    activeChatRef.current = updated
    setActiveChat(updated)
    // Announce the public counter; this is not a key.
    setStatusMessage(`Encrypted chat with ${updated.peerUsername} is ready (epoch ${nextEpoch}).`)
  }

  // Reconcile a just-sent message's temporary local id with the server's real row id
  // once its "accepted" ack arrives, so later edit/delete calls have something to
  // target. Edits reuse an already-reconciled id and do not enqueue here.
  function handleAccepted(serverId: string) {
    const localId = pendingSendQueueRef.current.shift()
    // No queued send means this ack is not for an optimistic bubble.
    if (localId === undefined) {
      // Ignore a stray accepted frame.
      return
    }
    // Remember the mapping so a slow classify() result still finds the row.
    acceptedIdsRef.current.set(localId, serverId)
    // Server reused the optimistic id; the bubble already has the right key.
    if (localId === serverId) {
      // Nothing to rename in React state or in the banner cache.
      return
    }
    // Follow the id change in the banner cache so reload looks up the server UUID.
    const username = usernameRef.current
    // Skip storage when ChatScreen is in the logout render.
    if (username) {
      // Move the pending-id row onto the history id.
      renameCachedBanner(username, localId, serverId)
    }
    setMessages((existing) =>
      existing.map((message) =>
        message.id === localId ? { ...message, id: serverId, pending: false } : message,
      ),
    )
  }

  // Open (or re-open) the ciphertext relay socket for one conversation, wiring every
  // callback this screen reacts to. Shared by the initial connect in openConversation
  // and by the auth-expiry reconnect in handleSocketClose so both paths stay identical.
  function openSocketForConversation(conversationId: string, accessToken: string): ChatSocket {
    return connectChatSocket(conversationId, accessToken, {
      onEnvelope: handleIncomingEnvelope,
      onAccepted: handleAccepted,
      onMessageDeleted: handleMessageDeleted,
      onProtocolError: (detail) => {
        setErrorMessage(detail)
      },
      onClose: (code) => {
        void handleSocketClose(conversationId, code)
      },
      onTyping: (userId, isTyping) => {
        if (activeChatRef.current?.peerId === userId) {
          setPeerTyping(isTyping)
        }
      },
      onPresence: (userId, online) => {
        if (activeChatRef.current?.peerId === userId) {
          setPeerOnline(online)
        }
      },
      onEpoch: (currentEpoch) => {
        // Re-derive the next encrypt subkey only; do not clear the composer.
        applyCurrentEpoch(currentEpoch)
      },
    })
  }

  // React to the relay socket closing. A WebSocket can only authenticate once, at
  // connect time (browsers cannot attach an Authorization header to the upgrade
  // request), so a token that expires mid-conversation cannot be refreshed in place
  // the way a REST call can — the socket simply gets closed with 4401 by the server.
  // Recover by refreshing the token pair once and reopening the same conversation's
  // socket, so a long-lived chat tab does not silently go "disconnected" every time
  // the access token's 15-minute lifetime elapses.
  async function handleSocketClose(conversationId: string, code: number) {
    // A contact switch or logout may have already torn this socket down on purpose;
    // only reconnect if this is still the conversation the user is looking at.
    if (activeChatRef.current?.conversationId !== conversationId) {
      return
    }
    if (code !== WS_CLOSE_UNAUTHORIZED) {
      setStatusMessage('Disconnected from the encrypted relay.')
      setPeerOnline(false)
      setPeerTyping(false)
      return
    }
    try {
      const rotated = await authorizedRequest.refreshAndRotate()
      socketRef.current = openSocketForConversation(conversationId, rotated.accessToken)
      setStatusMessage('Reconnected to the encrypted relay.')
    } catch {
      // A failed refresh here means the session is genuinely over; match the
      // existing behavior rather than looping reconnect attempts forever.
      setStatusMessage('Your session expired. Please log in again.')
      setPeerOnline(false)
      setPeerTyping(false)
    }
  }

  // Look up the peer, derive session keys, load scoped history, and open the ciphertext relay.
  async function openConversation(peerUsername: string) {
    if (!session) {
      // See the hook-level comment above: ChatScreen never actually calls this while
      // signed out, but the guard keeps this function safe to call unconditionally.
      return
    }
    // Bind to a local const so TypeScript keeps `session` narrowed to non-null for
    // the rest of this async function (narrowing does not reliably persist on a
    // captured outer parameter across every subsequent `await`).
    const currentSession = session
    setIsConnecting(true)
    setErrorMessage(null)
    setPeerTyping(false)
    setPeerOnline(false)
    setStatusMessage(`Looking up ${peerUsername}…`)
    socketRef.current?.close()
    socketRef.current = null
    pendingSendQueueRef.current = []
    // Drop optimistic-id mappings for the conversation we are leaving.
    acceptedIdsRef.current = new Map()
    setMessages([])
    setActiveChat(null)

    try {
      await initializeSodium()
      const peerKey = await authorizedRequest.request((accessToken) =>
        fetchPublicKey(accessToken, peerUsername),
      )
      const conversation = await authorizedRequest.request((accessToken) =>
        startOrFetchConversation(accessToken, peerUsername),
      )
      const epoch = await authorizedRequest.request((accessToken) =>
        fetchConversationEpoch(accessToken, conversation.id),
      )
      const sessionKeys = deriveSessionKeys(
        { keys: currentSession.identityKeyPair, username: currentSession.username },
        { publicKey: decodeBase64(peerKey.publicKey), username: peerKey.username },
      )
      const nextChat: ActiveChat = {
        conversationId: conversation.id,
        peerUsername: conversation.peer.username,
        selfId: conversation.self.id,
        peerId: conversation.peer.id,
        currentEpoch: epoch.currentEpoch,
        sessionKeys,
      }
      activeChatRef.current = nextChat
      setActiveChat(nextChat)
      const history = await authorizedRequest.request((accessToken) =>
        fetchConversationMessages(accessToken, conversation.id),
      )
      const hydrated: ChatMessage[] = history.map((envelope) => decryptEnvelope(envelope, nextChat))
      setMessages(hydrated)
      for (const row of hydrated) {
        if (needsClassify(row) && row.plaintext) {
          classifyAndFlag(row.id, row.plaintext, row.revision)
        }
      }
      socketRef.current = openSocketForConversation(conversation.id, currentSession.accessToken)
      setStatusMessage(
        `Encrypted chat with ${conversation.peer.username} is ready (epoch ${epoch.currentEpoch}).`,
      )
    } catch (error) {
      const message =
        error instanceof KeysApiError || error instanceof ConversationsApiError
          ? error.message
          : 'Could not start an encrypted chat. Please try again shortly.'
      setErrorMessage(message)
      setStatusMessage('Add a contact to start an encrypted chat.')
      setActiveChat(null)
    } finally {
      setIsConnecting(false)
    }
  }

  // Encrypt the draft with XChaCha20-Poly1305 + AD, then relay ciphertext only. Every
  // new send is v2 (carries a fresh client-chosen message identity at revision 0) so
  // it can later be edited; v1 remains only for decrypting history sent before
  // editing existed.
  function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const current = activeChatRef.current
    const socket = socketRef.current
    const plaintext = draft.trim()
    if (!current || !socket || !plaintext) {
      return
    }
    socket.sendTyping(false)
    if (typingTimeoutRef.current !== null) {
      window.clearTimeout(typingTimeoutRef.current)
      typingTimeoutRef.current = null
    }
    const clientMessageId = crypto.randomUUID()
    const envelope = encryptMessage(plaintext, current.sessionKeys.transmitKey, current.currentEpoch, {
      conversationId: current.conversationId,
      senderId: current.selfId,
      messageId: clientMessageId,
      revision: 0,
    })
    socket.sendEnvelope(
      envelope,
      { conversationId: current.conversationId, senderId: current.selfId },
      { messageId: clientMessageId, revision: 0 },
    )
    const localId = crypto.randomUUID()
    pendingSendQueueRef.current.push(localId)
    setMessages((existing) => [
      ...existing,
      {
        id: localId,
        direction: 'sent',
        plaintext,
        verificationFailed: false,
        scamWarning: false,
        clientMessageId,
        revision: 0,
        editedAt: null,
        pending: true,
      },
    ])
    classifyAndFlag(localId, plaintext, 0)
    setDraft('')
  }

  // Re-encrypt and resend an own message with the same message identity and an
  // advanced revision. Guarded so this can only ever target a message this account
  // actually sent, already reconciled with a real row id, not already verification-
  // failed, and originally sent as v2 (a v1 send predates editing and cannot gain an
  // identity retroactively without becoming indistinguishable from a new message).
  async function handleEditMessage(id: string, newPlaintext: string): Promise<void> {
    const current = activeChatRef.current
    const socket = socketRef.current
    const trimmed = newPlaintext.trim()
    if (!current || !socket || !trimmed) {
      return
    }
    const target = messages.find((message) => message.id === id)
    if (
      !target ||
      target.direction !== 'sent' ||
      target.pending ||
      target.verificationFailed ||
      target.clientMessageId === null
    ) {
      return
    }
    const nextRevision = target.revision + 1
    const envelope = encryptMessage(trimmed, current.sessionKeys.transmitKey, current.currentEpoch, {
      conversationId: current.conversationId,
      senderId: current.selfId,
      messageId: target.clientMessageId,
      revision: nextRevision,
    })
    socket.sendEnvelope(
      envelope,
      { conversationId: current.conversationId, senderId: current.selfId },
      { messageId: target.clientMessageId, revision: nextRevision },
    )
    // Optimistically apply the edit; the row's id never changes (see handleAccepted's
    // docstring — only a brand-new send needs id reconciliation).
    setMessages((existing) =>
      existing.map((message) =>
        message.id === id
          ? { ...message, plaintext: trimmed, revision: nextRevision, editedAt: new Date().toISOString() }
          : message,
      ),
    )
    // Re-score the new plaintext; a previous scam banner must be allowed to clear.
    classifyAndFlag(id, trimmed, nextRevision)
  }

  // Hard-delete an own message for every participant. Guarded to the sender's own,
  // already-reconciled, non-verification-failed messages; the peer is notified over
  // the existing WebSocket (message_deleted), not by this call's response.
  async function handleDeleteMessage(id: string): Promise<void> {
    const current = activeChatRef.current
    if (!current) {
      return
    }
    const target = messages.find((message) => message.id === id)
    if (!target || target.direction !== 'sent' || target.pending) {
      return
    }
    try {
      await authorizedRequest.request((accessToken) =>
        deleteConversationMessage(accessToken, current.conversationId, id),
      )
      setMessages((existing) => existing.filter((message) => message.id !== id))
    } catch (error) {
      const message =
        error instanceof ConversationsApiError
          ? error.message
          : 'Could not delete that message. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  // Hide a message (own or the peer's) from this account's own future history only.
  // The peer's copy, and the peer's own history, are never affected.
  async function handleHideMessage(id: string): Promise<void> {
    const current = activeChatRef.current
    if (!current) {
      return
    }
    const target = messages.find((message) => message.id === id)
    if (!target || target.pending) {
      return
    }
    try {
      await authorizedRequest.request((accessToken) =>
        hideConversationMessage(accessToken, current.conversationId, id),
      )
      setMessages((existing) => existing.filter((message) => message.id !== id))
    } catch (error) {
      const message =
        error instanceof ConversationsApiError
          ? error.message
          : 'Could not hide that message. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  // Update the draft and send typing metadata without including the draft text.
  function handleDraftChange(value: string) {
    setDraft(value)
    const socket = socketRef.current
    if (!socket || !activeChatRef.current) {
      return
    }
    if (value.trim()) {
      socket.sendTyping(true)
      if (typingTimeoutRef.current !== null) {
        window.clearTimeout(typingTimeoutRef.current)
      }
      typingTimeoutRef.current = window.setTimeout(() => {
        socket.sendTyping(false)
        typingTimeoutRef.current = null
      }, 1500)
    } else {
      socket.sendTyping(false)
    }
  }

  return {
    activeChat,
    messages,
    draft,
    isConnecting,
    peerOnline,
    peerTyping,
    openConversation,
    handleSend,
    handleDraftChange,
    handleEditMessage,
    handleDeleteMessage,
    handleHideMessage,
    handleClearLocalTranscript() {
      // Forget pending send ids so a later "accepted" ack cannot rename a gone row.
      pendingSendQueueRef.current = []
      // Drop optimistic-id mappings for the conversation we are leaving.
      acceptedIdsRef.current = new Map()
      setMessages([])
    },
    closeConversation() {
      // Drop the relay so a blocked peer cannot keep delivering envelopes here.
      socketRef.current?.close()
      socketRef.current = null
      // Forget pending ids the same way a local-clear does.
      pendingSendQueueRef.current = []
      // Drop optimistic-id mappings for the conversation we are leaving.
      acceptedIdsRef.current = new Map()
      // Stop treating this pairing as the open conversation.
      activeChatRef.current = null
      setActiveChat(null)
      setMessages([])
      setPeerOnline(false)
      setPeerTyping(false)
      setDraft('')
    },
  }
}
