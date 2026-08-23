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
    if (parsed && typeof parsed === 'object' && 'type' in parsed) {
      const type = (parsed as { type: unknown }).type
      if (type === 'error' && 'detail' in parsed && typeof (parsed as { detail: unknown }).detail === 'string') {
        handlers.onProtocolError((parsed as { detail: string }).detail)
        return
      }
      if (type === 'accepted') {
        // The sender already rendered plaintext locally; the id is optional metadata.
        return
      }
      if (type === 'typing') {
        const typingFrame = parsed as { user_id?: unknown; is_typing?: unknown }
        if (typeof typingFrame.user_id === 'string' && typeof typingFrame.is_typing === 'boolean') {
          handlers.onTyping?.(typingFrame.user_id, typingFrame.is_typing)
        }
        return
      }
      if (type === 'presence') {
        const presenceFrame = parsed as { user_id?: unknown; online?: unknown }
        if (typeof presenceFrame.user_id === 'string' && typeof presenceFrame.online === 'boolean') {
          handlers.onPresence?.(presenceFrame.user_id, presenceFrame.online)
        }
        return
      }
    }
    const envelope = parseRelayedEnvelope(parsed)
    if (envelope === null) {
      handlers.onProtocolError('Received an unreadable ciphertext envelope.')
      return
    }
    handlers.onEnvelope(envelope)
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
