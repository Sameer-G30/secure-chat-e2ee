// Import React state and lifecycle hooks used to drive the live conversation.
import { useEffect, useRef, useState } from 'react'
// Import the form-event type separately; verbatimModuleSyntax requires type-only imports.
import type { FormEvent } from 'react'

// Import the public-key lookup used before session-key derivation.
import { fetchPublicKey, KeysApiError } from '../api/keysClient'
// Import conversation create/fetch and the spec §6.4 epoch reader.
import {
  ConversationsApiError,
  fetchConversationEpoch,
  startOrFetchConversation,
} from '../api/conversationsClient'
// Import the authenticated ciphertext-only WebSocket wrapper.
import { connectChatSocket } from '../api/chatSocket'
import type { ChatSocket, RelayedEnvelope } from '../api/chatSocket'
// Import the session context so this screen can show identity and offer logout.
import { useAuth } from '../context/AuthContext'
// Import the production crypto helpers used to encrypt before send and verify on receive.
import {
  decodeBase64,
  decryptMessage,
  deriveSessionKeys,
  encryptMessage,
  initializeSodium,
} from '../crypto/keyExchange'
import type { DirectionalSessionKeys } from '../crypto/keyExchange'
// Import the on-device scam classifier (TF-IDF Best eager; DistilBERT / BiLSTM opt-in).
import {
  classifyVerifiedPlaintext,
  disableDistilbertOptIn,
  disableLstmOptIn,
  enableDistilbertOptIn,
  enableLstmOptIn,
  ensureChatDefaultClassifier,
} from '../ml/scamClassifier'
import type { ChatHeavyPreference } from '../ml/types'

// Describe one bubble in the in-memory message list (Slice 4 has no history pagination).
interface ChatMessage {
  // Identify the row for React reconciliation; server ids are used when available.
  id: string
  // Distinguish sent (purple) from received (gray) bubbles using the legacy palette.
  direction: 'sent' | 'received'
  // Carry verified plaintext, or null when authentication failed.
  plaintext: string | null
  // Signal that decrypt+verify failed; the UI must not invent corrupted text.
  verificationFailed: boolean
  // Non-blocking banner after local classification of verified plaintext.
  scamWarning: boolean
}

// Describe the live conversation this tab has derived keys for.
interface ActiveChat {
  // Bind AEAD associated data and WebSocket routing to this conversation UUID.
  conversationId: string
  // Display the peer handle in the header.
  peerUsername: string
  // Bind AEAD associated data to this tab's user UUID.
  selfId: string
  // Identify the peer UUID for received-envelope associated data.
  peerId: string
  // Pass the server's current epoch into encrypt; decrypt uses the envelope's keyEpoch.
  currentEpoch: number
  // Hold directional crypto_kx keys derived locally; never sent to the server.
  sessionKeys: DirectionalSessionKeys
}

// Render the Slice 4 two-tab encrypted conversation shell.
export function ChatScreen() {
  const { session, logout } = useAuth()

  // Hold the peer username the user typed before starting a conversation.
  const [peerInput, setPeerInput] = useState('')
  // Hold a status string announced to assistive technology.
  const [statusMessage, setStatusMessage] = useState('Enter a peer username to start an encrypted chat.')
  // Hold a form-level error distinct from per-message verification failures.
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  // Hold whether a conversation start is in flight so the form cannot double-submit.
  const [isConnecting, setIsConnecting] = useState(false)
  // Hold the in-memory transcript for this tab only (no server history UI in this slice).
  const [messages, setMessages] = useState<ChatMessage[]>([])
  // Hold the composer text until send encrypts and clears it.
  const [draft, setDraft] = useState('')
  // Hold the derived conversation so encrypt/decrypt can read ids and epoch.
  const [activeChat, setActiveChat] = useState<ActiveChat | null>(null)
  // Hold whether the operator opted into lazy-loaded DistilBERT (A6).
  const [useDistilbert, setUseDistilbert] = useState(false)
  // Hold whether the operator opted into lazy-loaded Word BiLSTM Best.
  const [useLstm, setUseLstm] = useState(false)
  // Hold classifier load status so toggles can show a failure without blocking chat.
  const [classifierStatus, setClassifierStatus] = useState<string | null>(null)

  // Keep the live socket off React state so reconnects do not re-render on every frame.
  const socketRef = useRef<ChatSocket | null>(null)
  // Keep the latest active chat in a ref so socket callbacks never close over a stale epoch.
  const activeChatRef = useRef<ActiveChat | null>(null)
  // Keep the selected classifier in a ref so incoming envelopes use the current model.
  const heavyPreferenceRef = useRef<ChatHeavyPreference>('tfidf')

  // Mirror active chat into the ref whenever React state changes.
  useEffect(() => {
    activeChatRef.current = activeChat
  }, [activeChat])

  // Mirror the model toggles into the ref used by socket callbacks.
  useEffect(() => {
    heavyPreferenceRef.current = useDistilbert ? 'distilbert' : useLstm ? 'lstm' : 'tfidf'
  }, [useDistilbert, useLstm])

  // Eager-load TF-IDF Best once ChatScreen mounts (A5).
  useEffect(() => {
    void ensureChatDefaultClassifier().catch(() => {
      // Missing ONNX export must not block the encrypted conversation.
    })
  }, [])

  // Close any open relay socket when this screen unmounts (logout or session end).
  useEffect(() => {
    return () => {
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [])

  // This screen is only ever rendered while a session exists (see App.tsx routing).
  if (!session) {
    return null
  }
  const currentSession = session

  // Classify verified plaintext locally and attach a non-blocking banner.
  function classifyAndFlag(messageId: string, plaintext: string) {
    // Skip empty strings; they have no scam signal and would still hit ORT.
    if (!plaintext) {
      // Leave scamWarning false.
      return
    }
    // Run after decrypt/send so the bubble appears before WASM returns.
    void classifyVerifiedPlaintext(plaintext, heavyPreferenceRef.current).then((result) => {
      // A missing export or WASM abort leaves the message unwarned.
      if (result === null || !result.warned) {
        // No banner to attach.
        return
      }
      // Flip only this row's warning flag; never hide or delete the text.
      setMessages((existing) =>
        existing.map((message) =>
          message.id === messageId ? { ...message, scamWarning: true } : message,
        ),
      )
    })
  }

  // Decrypt a peer envelope, or record a verification failure without rendering garbage.
  function handleIncomingEnvelope(envelope: RelayedEnvelope) {
    const current = activeChatRef.current
    if (current === null) {
      return
    }
    if (envelope.senderId === current.selfId) {
      // The server does not echo to the sender; ignore anyway so we never decrypt our own Tx.
      return
    }
    if (envelope.senderId !== current.peerId) {
      setMessages((existing) => [
        ...existing,
        {
          id: envelope.id,
          direction: 'received',
          plaintext: null,
          verificationFailed: true,
          scamWarning: false,
        },
      ])
      return
    }
    try {
      const plaintext = decryptMessage(
        {
          ciphertext: envelope.ciphertext,
          nonce: envelope.nonce,
          keyEpoch: envelope.keyEpoch,
        },
        current.sessionKeys.receiveKey,
        {
          conversationId: current.conversationId,
          senderId: envelope.senderId,
        },
      )
      setMessages((existing) => [
        ...existing,
        {
          id: envelope.id,
          direction: 'received',
          plaintext,
          verificationFailed: false,
          scamWarning: false,
        },
      ])
      classifyAndFlag(envelope.id, plaintext)
    } catch {
      setMessages((existing) => [
        ...existing,
        {
          id: envelope.id,
          direction: 'received',
          plaintext: null,
          verificationFailed: true,
          scamWarning: false,
        },
      ])
    }
  }

  // Look up the peer, derive session keys, fetch epoch, and open the ciphertext relay.
  async function handleStartChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const peerUsername = peerInput.trim()
    if (!peerUsername) {
      setErrorMessage('Enter the username you want to chat with.')
      return
    }
    if (peerUsername === currentSession.username) {
      setErrorMessage('You cannot start an encrypted conversation with yourself.')
      return
    }

    setIsConnecting(true)
    setErrorMessage(null)
    setStatusMessage(`Looking up ${peerUsername}…`)
    socketRef.current?.close()
    socketRef.current = null
    setMessages([])
    setActiveChat(null)

    try {
      await initializeSodium()
      const peerKey = await fetchPublicKey(currentSession.accessToken, peerUsername)
      const conversation = await startOrFetchConversation(currentSession.accessToken, peerUsername)
      const epoch = await fetchConversationEpoch(currentSession.accessToken, conversation.id)
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
      socketRef.current = connectChatSocket(conversation.id, currentSession.accessToken, {
        onEnvelope: handleIncomingEnvelope,
        onProtocolError: (detail) => {
          setErrorMessage(detail)
        },
        onClose: () => {
          setStatusMessage('Disconnected from the encrypted relay.')
        },
      })
      setStatusMessage(
        `Encrypted chat with ${conversation.peer.username} is ready (epoch ${epoch.currentEpoch}).`,
      )
    } catch (error) {
      const message =
        error instanceof KeysApiError || error instanceof ConversationsApiError
          ? error.message
          : 'Could not start an encrypted chat. Please try again shortly.'
      setErrorMessage(message)
      setStatusMessage('Enter a peer username to start an encrypted chat.')
      setActiveChat(null)
    } finally {
      setIsConnecting(false)
    }
  }

  // Encrypt the draft with XChaCha20-Poly1305 + AD, then relay ciphertext only.
  function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const current = activeChatRef.current
    const socket = socketRef.current
    const plaintext = draft.trim()
    if (!current || !socket || !plaintext) {
      return
    }
    const envelope = encryptMessage(
      plaintext,
      current.sessionKeys.transmitKey,
      current.currentEpoch,
      {
        conversationId: current.conversationId,
        senderId: current.selfId,
      },
    )
    socket.sendEnvelope(envelope, {
      conversationId: current.conversationId,
      senderId: current.selfId,
    })
    const localId = crypto.randomUUID()
    setMessages((existing) => [
      ...existing,
      {
        id: localId,
        direction: 'sent',
        plaintext,
        verificationFailed: false,
        scamWarning: false,
      },
    ])
    classifyAndFlag(localId, plaintext)
    setDraft('')
  }

  return (
    <main className="chat-screen">
      <header className="chat-header">
        <div>
          <h1>Secure Chat</h1>
          <p>
            Signed in as <strong>{session.username}</strong>
            {activeChat ? (
              <>
                {' '}
                · chatting with <strong>{activeChat.peerUsername}</strong>
              </>
            ) : null}
          </p>
        </div>
        <div className="chat-header-actions">
          <div className="chat-model-toggles">
            <label className="chat-model-toggle">
              <input
                type="checkbox"
                checked={useDistilbert}
                onChange={(event) => {
                  const enabled = event.target.checked
                  if (enabled) {
                    setUseLstm(false)
                    setUseDistilbert(true)
                    setClassifierStatus('Loading DistilBERT on this device…')
                    void enableDistilbertOptIn()
                      .then(() => {
                        setClassifierStatus(
                          'DistilBERT is classifying on this device (not sent to the server).',
                        )
                      })
                      .catch((caught: unknown) => {
                        setUseDistilbert(false)
                        setClassifierStatus(
                          caught instanceof Error
                            ? `DistilBERT failed to load: ${caught.message}`
                            : 'DistilBERT failed to load; staying on TF-IDF Best.',
                        )
                      })
                  } else {
                    setUseDistilbert(false)
                    void disableDistilbertOptIn()
                    setClassifierStatus('Using the on-device TF-IDF Best classifier.')
                  }
                }}
              />
              Use DistilBERT (large download)
            </label>
            <label className="chat-model-toggle">
              <input
                type="checkbox"
                checked={useLstm}
                onChange={(event) => {
                  const enabled = event.target.checked
                  if (enabled) {
                    setUseDistilbert(false)
                    setUseLstm(true)
                    setClassifierStatus('Loading Word BiLSTM Best on this device…')
                    void enableLstmOptIn()
                      .then(() => {
                        setClassifierStatus(
                          'Word BiLSTM Best is classifying on this device (not sent to the server).',
                        )
                      })
                      .catch((caught: unknown) => {
                        setUseLstm(false)
                        setClassifierStatus(
                          caught instanceof Error
                            ? `Word BiLSTM Best failed to load: ${caught.message}`
                            : 'Word BiLSTM Best failed to load; staying on TF-IDF Best.',
                        )
                      })
                  } else {
                    setUseLstm(false)
                    void disableLstmOptIn()
                    setClassifierStatus('Using the on-device TF-IDF Best classifier.')
                  }
                }}
              />
              Use Word BiLSTM Best
            </label>
          </div>
          <button type="button" className="text-button" onClick={() => void logout()}>
            Log out
          </button>
        </div>
      </header>

      <form className="chat-peer-form" onSubmit={(event) => void handleStartChat(event)}>
        <div className="input-group">
          <label htmlFor="peer-username">Peer username</label>
          <input
            id="peer-username"
            name="peerUsername"
            type="text"
            autoComplete="username"
            value={peerInput}
            onChange={(event) => setPeerInput(event.target.value)}
            required
          />
        </div>
        <button className="primary-button chat-start-button" type="submit" disabled={isConnecting}>
          {isConnecting ? 'Connecting…' : 'Start encrypted chat'}
        </button>
      </form>

      <p className="chat-status" role="status">
        {statusMessage}
      </p>
      {errorMessage ? (
        <p className="auth-feedback auth-feedback-error" role="alert">
          {errorMessage}
        </p>
      ) : null}

      {classifierStatus ? (
        <p className="chat-status" role="status">
          {classifierStatus}
        </p>
      ) : null}

      <section className="chat-transcript" aria-label="Encrypted messages">
        {messages.length === 0 ? (
          <p className="chat-empty">No messages yet. Ciphertext is relayed through the server; plaintext stays on this device.</p>
        ) : (
          <ul className="chat-message-list">
            {messages.map((message) => (
              <li
                key={message.id}
                className={
                  message.verificationFailed
                    ? 'chat-bubble chat-bubble-failed'
                    : message.direction === 'sent'
                      ? 'chat-bubble chat-bubble-sent'
                      : 'chat-bubble chat-bubble-received'
                }
              >
                {message.verificationFailed ? (
                  <span role="alert">message failed verification</span>
                ) : (
                  <>
                    {message.scamWarning ? (
                      <p className="scam-banner" role="status">
                        This message shows signs of a scam
                      </p>
                    ) : null}
                    {message.plaintext}
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <form className="chat-composer" onSubmit={handleSend}>
        <label className="visually-hidden" htmlFor="chat-draft">
          Message
        </label>
        <input
          id="chat-draft"
          name="draft"
          type="text"
          autoComplete="off"
          placeholder="Type a message"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          disabled={!activeChat}
        />
        <button className="primary-button chat-send-button" type="submit" disabled={!activeChat || !draft.trim()}>
          Send
        </button>
      </form>

      <p className="slice-note" role="status">
        Slice 6: two browser tabs still hold a real encrypted conversation through the server.
        After decrypt (and on send), this tab classifies plaintext locally with ONNX Runtime Web.
        The eager default is TF-IDF Best (10k terms, A5). DistilBERT default and Word BiLSTM Best
        are lazy opt-in toggles (one heavy graph at a time). DistilBERT encodes real token length
        only and runs in an ORT Web Worker so the composer stays responsive.
        Contacts, history pagination, typing/presence, and dark mode stay later slices.
      </p>
    </main>
  )
}
