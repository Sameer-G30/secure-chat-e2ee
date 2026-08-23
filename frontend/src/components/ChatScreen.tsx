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
  fetchConversationMessages,
  startOrFetchConversation,
} from '../api/conversationsClient'
// Import the authenticated ciphertext-only WebSocket wrapper.
import { connectChatSocket } from '../api/chatSocket'
import type { ChatSocket, RelayedEnvelope } from '../api/chatSocket'
// Import the server-side address book so contacts never live in localStorage.
import { addContact, ContactsApiError, listContacts } from '../api/contactsClient'
import type { ContactRecord } from '../api/contactsClient'
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
// Import per-username theme helpers; tokens stay out of localStorage.
import { applyTheme, clearTheme, loadTheme, saveTheme } from '../theme'
import type { ThemeName } from '../theme'

// Build a one-letter avatar label from a username.
function initialsFromUsername(username: string): string {
  // Trim so a padded handle still yields a letter.
  const trimmed = username.trim()
  // Fall back to a question mark when the handle is empty.
  if (!trimmed) {
    return '?'
  }
  // Use the first character so the avatar matches the legacy chrome.
  return trimmed.slice(0, 1).toUpperCase()
}

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

  // Hold the add-contact field until the owner saves a handle on the server.
  const [peerInput, setPeerInput] = useState('')
  // Hold the server-side address book loaded after login.
  const [contacts, setContacts] = useState<ContactRecord[]>([])
  // Hold a status string announced to assistive technology.
  const [statusMessage, setStatusMessage] = useState(
    'Add a contact to start an encrypted chat.',
  )
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
  // Hold the per-user light/dark preference applied to the document.
  const [theme, setTheme] = useState<ThemeName>('light')
  // Hold whether the open peer currently has a live socket (metadata, not a secret).
  const [peerOnline, setPeerOnline] = useState(false)
  // Hold whether the open peer is currently typing (metadata, never draft text).
  const [peerTyping, setPeerTyping] = useState(false)

  // Keep the live socket off React state so reconnects do not re-render on every frame.
  const socketRef = useRef<ChatSocket | null>(null)
  // Keep the latest active chat in a ref so socket callbacks never close over a stale epoch.
  const activeChatRef = useRef<ActiveChat | null>(null)
  // Keep the selected classifier in a ref so incoming envelopes use the current model.
  const heavyPreferenceRef = useRef<ChatHeavyPreference>('tfidf')
  // Remember envelope ids already rendered so history and live frames do not duplicate.
  const seenIdsRef = useRef<Set<string>>(new Set())
  // Debounce the typing-false frame so keystrokes do not flood the metadata channel.
  const typingTimeoutRef = useRef<number | null>(null)

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
      if (typingTimeoutRef.current !== null) {
        window.clearTimeout(typingTimeoutRef.current)
      }
    }
  }, [])

  // Load this username's theme, then restore the default palette when the session ends.
  useEffect(() => {
    if (!session) {
      clearTheme()
      return
    }
    const next = loadTheme(session.username)
    setTheme(next)
    applyTheme(next)
    return () => {
      clearTheme()
    }
  }, [session])

  // Load the server-side address book once a session exists.
  useEffect(() => {
    if (!session) {
      return
    }
    let cancelled = false
    void listContacts(session.accessToken)
      .then((rows) => {
        if (!cancelled) {
          setContacts(rows)
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          const message =
            caught instanceof ContactsApiError
              ? caught.message
              : 'Could not load contacts. Please try again shortly.'
          setErrorMessage(message)
        }
      })
    return () => {
      cancelled = true
    }
  }, [session])

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
    if (key === null) {
      return {
        id: envelope.id,
        direction: 'received',
        plaintext: null,
        verificationFailed: true,
        scamWarning: false,
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
        },
      )
      return {
        id: envelope.id,
        direction,
        plaintext,
        verificationFailed: false,
        scamWarning: false,
      }
    } catch {
      return {
        id: envelope.id,
        direction,
        plaintext: null,
        verificationFailed: true,
        scamWarning: false,
      }
    }
  }

  // Decrypt a live peer envelope, skipping ids already shown from history.
  function handleIncomingEnvelope(envelope: RelayedEnvelope) {
    const current = activeChatRef.current
    if (current === null) {
      return
    }
    if (seenIdsRef.current.has(envelope.id)) {
      return
    }
    seenIdsRef.current.add(envelope.id)
    if (envelope.senderId === current.selfId) {
      return
    }
    const row = decryptEnvelope(envelope, current)
    setMessages((existing) => [...existing, row])
    if (!row.verificationFailed && row.plaintext) {
      classifyAndFlag(row.id, row.plaintext)
    }
  }

  // Look up the peer, derive session keys, load scoped history, and open the ciphertext relay.
  async function openConversation(peerUsername: string) {
    setIsConnecting(true)
    setErrorMessage(null)
    setPeerTyping(false)
    setPeerOnline(false)
    setStatusMessage(`Looking up ${peerUsername}…`)
    socketRef.current?.close()
    socketRef.current = null
    seenIdsRef.current = new Set()
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
      const history = await fetchConversationMessages(
        currentSession.accessToken,
        conversation.id,
      )
      const hydrated: ChatMessage[] = []
      for (const envelope of history) {
        seenIdsRef.current.add(envelope.id)
        hydrated.push(decryptEnvelope(envelope, nextChat))
      }
      setMessages(hydrated)
      for (const row of hydrated) {
        if (!row.verificationFailed && row.plaintext) {
          classifyAndFlag(row.id, row.plaintext)
        }
      }
      socketRef.current = connectChatSocket(conversation.id, currentSession.accessToken, {
        onEnvelope: handleIncomingEnvelope,
        onProtocolError: (detail) => {
          setErrorMessage(detail)
        },
        onClose: () => {
          setStatusMessage('Disconnected from the encrypted relay.')
          setPeerOnline(false)
          setPeerTyping(false)
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
      setStatusMessage('Add a contact to start an encrypted chat.')
      setActiveChat(null)
    } finally {
      setIsConnecting(false)
    }
  }

  // Save a handle on the server-side address book, then open that conversation.
  async function handleAddContact(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const peerUsername = peerInput.trim()
    if (!peerUsername) {
      setErrorMessage('Enter the username you want to chat with.')
      return
    }
    if (peerUsername === currentSession.username) {
      setErrorMessage('You cannot add yourself as a contact.')
      return
    }
    setErrorMessage(null)
    setStatusMessage(`Adding ${peerUsername}…`)
    try {
      const saved = await addContact(currentSession.accessToken, peerUsername)
      setContacts((existing) => {
        if (existing.some((row) => row.id === saved.id)) {
          return existing
        }
        return [saved, ...existing]
      })
      setPeerInput('')
      await openConversation(saved.username)
    } catch (caught: unknown) {
      const message =
        caught instanceof ContactsApiError
          ? caught.message
          : 'Could not add that contact. Please try again shortly.'
      setErrorMessage(message)
      setStatusMessage('Add a contact to start an encrypted chat.')
    }
  }

  // Open an existing contact without creating a duplicate address-book row.
  function handleSelectContact(username: string) {
    void openConversation(username)
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
    socket.sendTyping(false)
    if (typingTimeoutRef.current !== null) {
      window.clearTimeout(typingTimeoutRef.current)
      typingTimeoutRef.current = null
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

  // Persist the theme for this username only and apply it to the document.
  function handleToggleTheme() {
    const next: ThemeName = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    applyTheme(next)
    saveTheme(currentSession.username, next)
  }

  // Drop the per-user theme attribute, then clear the in-memory session.
  function handleLogout() {
    clearTheme()
    void logout()
  }

  return (
    <div className="chat-shell">
      <aside className="chat-sidebar" aria-label="Contacts">
        <div className="chat-sidebar-self">
          <span className="chat-avatar" aria-hidden="true">
            {initialsFromUsername(currentSession.username)}
          </span>
          <div>
            <p className="chat-sidebar-name">{currentSession.username}</p>
            <p className="chat-sidebar-hint">Signed in</p>
          </div>
        </div>
        <form className="chat-add-contact" onSubmit={(event) => void handleAddContact(event)}>
          <div className="input-group">
            <label htmlFor="add-contact">Add contact</label>
            <input
              id="add-contact"
              name="addContact"
              type="text"
              autoComplete="username"
              value={peerInput}
              onChange={(event) => setPeerInput(event.target.value)}
              required
            />
          </div>
          <button className="primary-button chat-add-button" type="submit" disabled={isConnecting}>
            {isConnecting ? 'Connecting…' : 'Add'}
          </button>
        </form>
        {contacts.length === 0 ? (
          <p className="chat-empty">No contacts yet. Add a username to start chatting.</p>
        ) : (
          <ul className="chat-contact-list">
            {contacts.map((contact) => (
              <li key={contact.id}>
                <button
                  type="button"
                  className={
                    activeChat?.peerId === contact.id
                      ? 'chat-contact-button chat-contact-button-active'
                      : 'chat-contact-button'
                  }
                  onClick={() => handleSelectContact(contact.username)}
                  disabled={isConnecting}
                >
                  <span className="chat-avatar chat-avatar-sm" aria-hidden="true">
                    {initialsFromUsername(contact.username)}
                  </span>
                  <span className="chat-contact-name">{contact.username}</span>
                  {activeChat?.peerId === contact.id ? (
                    <span
                      className={
                        peerOnline ? 'chat-online-dot chat-online-dot-on' : 'chat-online-dot'
                      }
                      aria-hidden="true"
                    />
                  ) : null}
                </button>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-identity">
            {activeChat ? (
              <span className="chat-avatar" aria-hidden="true">
                {initialsFromUsername(activeChat.peerUsername)}
              </span>
            ) : null}
            <div>
              <h1>Secure Chat</h1>
              <p>
                Signed in as <strong>{currentSession.username}</strong>
                {activeChat ? (
                  <>
                    {' '}
                    · chatting with <strong>{activeChat.peerUsername}</strong>{' '}
                    <span className={peerOnline ? 'chat-presence chat-presence-on' : 'chat-presence'}>
                      {peerOnline ? 'Online' : 'Offline'}
                    </span>
                  </>
                ) : null}
              </p>
            </div>
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
            <button type="button" className="text-button" onClick={handleToggleTheme}>
              {theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            </button>
            <button type="button" className="text-button" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </header>

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

        {peerTyping && activeChat ? (
          <p className="chat-typing" role="status">
            {activeChat.peerUsername} is typing…
          </p>
        ) : null}

        <section className="chat-transcript" aria-label="Encrypted messages">
          {messages.length === 0 ? (
            <p className="chat-empty">
              No messages yet. Ciphertext is relayed through the server; plaintext stays on this
              device.
            </p>
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
            onChange={(event) => handleDraftChange(event.target.value)}
            disabled={!activeChat}
          />
          <button
            className="primary-button chat-send-button"
            type="submit"
            disabled={!activeChat || !draft.trim()}
          >
            Send
          </button>
        </form>

        <p className="slice-note" role="status">
          Slice 7: contacts live on the server, not in localStorage. Opening a chat loads that
          conversation&apos;s ciphertext envelopes, then this tab decrypts and classifies them
          locally (TF-IDF Best eager; DistilBERT / Word BiLSTM Best are lazy XOR opt-ins). Typing
          and presence are metadata the server can see; draft text never leaves this device.
        </p>
      </main>
    </div>
  )
}
