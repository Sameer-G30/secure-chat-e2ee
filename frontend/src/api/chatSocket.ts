// Import the URL builder that puts the access token on the WebSocket query string.
import { conversationWebSocketUrl } from './conversationsClient'
// Import the ciphertext-only envelope type the crypto module already documents.
import type { EncryptedEnvelope } from '../crypto/keyExchange'
import { decodeBase64, encodeBase64 } from '../crypto/keyExchange'

// Describe one ciphertext frame the server fans out after persisting it.
export interface RelayedEnvelope {
  // Identify the persisted row so the UI can key the message list.
  id: string
  // Identify the conversation this envelope belongs to (AEAD associated data).
  conversationId: string
  // Identify the sender this envelope claims (AEAD associated data).
  senderId: string
  // Carry authenticated ciphertext bytes without plaintext.
  ciphertext: Uint8Array
  // Carry the unique public nonce required for decryption.
  nonce: Uint8Array
  // Identify which locally derived epoch key protects this envelope.
  keyEpoch: number
}

// Describe the callbacks a ChatScreen connection registers.
export interface ChatSocketHandlers {
  // Receive a peer envelope that still needs client-side decrypt+verify.
  onEnvelope: (envelope: RelayedEnvelope) => void
  // Surface a protocol-layer rejection without rendering ciphertext as text.
  onProtocolError: (detail: string) => void
  // Notify the UI when the socket closes so it can show a disconnected state.
  onClose: () => void
  // Receive a typing metadata event; the server never sees draft text.
  onTyping?: (userId: string, isTyping: boolean) => void
  // Receive a presence metadata event; this is not a secret.
  onPresence?: (userId: string, online: boolean) => void
  // Receive a non-secret epoch counter bump; clients re-derive the next encrypt subkey locally.
  onEpoch?: (currentEpoch: number) => void
}

// Describe the handle ChatScreen uses to send envelopes and tear the socket down.
export interface ChatSocket {
  // Send one locally encrypted envelope; the server must never see plaintext.
  sendEnvelope: (
    envelope: EncryptedEnvelope,
    routing: { conversationId: string; senderId: string },
  ) => void
  // Send a typing metadata frame without including the draft text.
  sendTyping: (isTyping: boolean) => void
  // Close the socket when the user leaves the conversation or logs out.
  close: () => void
}

// Narrow an unknown JSON value into a relayed envelope, or return null if it is not one.
export function parseRelayedEnvelope(value: unknown): RelayedEnvelope | null {
  if (!value || typeof value !== 'object') {
    return null
  }
  const frame = value as {
    type?: unknown
    id?: unknown
    conversation_id?: unknown
    sender_id?: unknown
    ciphertext?: unknown
    nonce?: unknown
    key_epoch?: unknown
  }
  if (frame.type !== 'envelope') {
    return null
  }
  if (
    typeof frame.id !== 'string' ||
    typeof frame.conversation_id !== 'string' ||
    typeof frame.sender_id !== 'string' ||
    typeof frame.ciphertext !== 'string' ||
    typeof frame.nonce !== 'string' ||
    typeof frame.key_epoch !== 'number'
  ) {
    return null
  }
  try {
    return {
      id: frame.id,
      conversationId: frame.conversation_id,
      senderId: frame.sender_id,
      ciphertext: decodeBase64(frame.ciphertext),
      nonce: decodeBase64(frame.nonce),
      keyEpoch: frame.key_epoch,
    }
  } catch {
    // Malformed base64 is treated as a dropped frame rather than corrupted plaintext.
    return null
  }
}

// Narrow an unknown JSON value into the non-secret epoch counter, or return null.
export function parseEpochFrame(value: unknown): number | null {
  // Reject primitives and null; an epoch frame is always a JSON object.
  if (!value || typeof value !== 'object') {
    // Callers treat null as "not an epoch frame".
    return null
  }
  // Read only the two fields the server is allowed to put on this metadata frame.
  const frame = value as { type?: unknown; current_epoch?: unknown }
  // Ignore envelopes, typing, presence, and acks.
  if (frame.type !== 'epoch') {
    // This is some other relay frame.
    return null
  }
  // Require a non-negative safe integer; the server never sends a key here.
  if (typeof frame.current_epoch !== 'number' || !Number.isSafeInteger(frame.current_epoch)) {
    // A malformed counter must not update encrypt state.
    return null
  }
  // Reject negative ids the KDF would also refuse.
  if (frame.current_epoch < 0) {
    // Keep the previous in-memory epoch.
    return null
  }
  // Return the public counter clients pass to deriveEpochKey on the next encrypt.
  return frame.current_epoch
}

// Dispatch one already-parsed relay JSON object to the matching ChatScreen callback.
export function handleRelayJson(parsed: unknown, handlers: ChatSocketHandlers): void {
  // Metadata frames are identified by a string type discriminator.
  if (parsed && typeof parsed === 'object' && 'type' in parsed) {
    // Read the discriminator once so each branch stays narrow.
    const type = (parsed as { type: unknown }).type
    // Protocol errors never include ciphertext or plaintext.
    if (type === 'error' && 'detail' in parsed && typeof (parsed as { detail: unknown }).detail === 'string') {
      // Surface the rejection in the chat status line.
      handlers.onProtocolError((parsed as { detail: string }).detail)
      // Stop; this is not an envelope.
      return
    }
    // Persist acks are optional metadata for the sender tab.
    if (type === 'accepted') {
      // The sender already rendered plaintext locally.
      return
    }
    // Typing frames must never carry draft text.
    if (type === 'typing') {
      // Narrow the metadata fields the UI actually uses.
      const typingFrame = parsed as { user_id?: unknown; is_typing?: unknown }
      // Ignore malformed typing metadata rather than treating it as ciphertext.
      if (typeof typingFrame.user_id === 'string' && typeof typingFrame.is_typing === 'boolean') {
        // Fan-out already excluded the sender on the server.
        handlers.onTyping?.(typingFrame.user_id, typingFrame.is_typing)
      }
      // Never fall through to envelope parsing.
      return
    }
    // Presence is online/offline metadata the server can already see.
    if (type === 'presence') {
      // Narrow the metadata fields the UI actually uses.
      const presenceFrame = parsed as { user_id?: unknown; online?: unknown }
      // Ignore malformed presence rather than showing an unreadable-envelope error.
      if (typeof presenceFrame.user_id === 'string' && typeof presenceFrame.online === 'boolean') {
        // Update the online dot for this peer.
        handlers.onPresence?.(presenceFrame.user_id, presenceFrame.online)
      }
      // Never fall through to envelope parsing.
      return
    }
    // Epoch bumps are a public counter; the server never sends a key.
    if (type === 'epoch') {
      // Parse the integer clients will use for the next encrypt only.
      const currentEpoch = parseEpochFrame(parsed)
      // A well-formed bump updates in-memory currentEpoch without touching draft text.
      if (currentEpoch !== null) {
        // Decrypt still uses each envelope's own key_epoch.
        handlers.onEpoch?.(currentEpoch)
      }
      // Always consume epoch frames so they are not reported as unreadable envelopes.
      return
    }
  }
  // Remaining frames must be ciphertext envelopes, or they are protocol noise.
  const envelope = parseRelayedEnvelope(parsed)
  // A missing type=envelope (or bad base64) is not recoverable plaintext.
  if (envelope === null) {
    // Tell the UI without inventing a body.
    handlers.onProtocolError('Received an unreadable ciphertext envelope.')
    // Stop before calling onEnvelope.
    return
  }
  // Hand the ciphertext to ChatScreen for local decrypt+verify.
  handlers.onEnvelope(envelope)
}

// Open an authenticated WebSocket to the ciphertext relay for one conversation.
export function connectChatSocket(
  conversationId: string,
  accessToken: string,
  handlers: ChatSocketHandlers,
): ChatSocket {
  const socket = new WebSocket(conversationWebSocketUrl(conversationId, accessToken))

  socket.addEventListener('message', (event: MessageEvent<string>) => {
    let parsed: unknown
    try {
      parsed = JSON.parse(event.data) as unknown
    } catch {
      handlers.onProtocolError('Received a malformed relay frame.')
      return
    }
    handleRelayJson(parsed, handlers)
  })

  socket.addEventListener('close', () => {
    handlers.onClose()
  })

  return {
    sendEnvelope(envelope, routing) {
      if (socket.readyState !== WebSocket.OPEN) {
        handlers.onProtocolError('The encrypted connection is not open yet.')
        return
      }
      socket.send(
        JSON.stringify({
          ciphertext: encodeBase64(envelope.ciphertext),
          nonce: encodeBase64(envelope.nonce),
          key_epoch: envelope.keyEpoch,
          conversation_id: routing.conversationId,
          sender_id: routing.senderId,
        }),
      )
    },
    sendTyping(isTyping) {
      if (socket.readyState !== WebSocket.OPEN) {
        return
      }
      socket.send(JSON.stringify({ type: 'typing', is_typing: isTyping }))
    },
    close() {
      if (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) {
        socket.close()
      }
    },
  }
}
