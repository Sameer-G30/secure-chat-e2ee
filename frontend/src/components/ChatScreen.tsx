// Import React's state/callback hooks used by the small amount of state this
// presentational shell still owns directly (shared status/error banners, the
// add-contact input, overlays, and conversation-start in-flight flag).
import { useCallback, useState } from 'react'
// Import the form-event type separately; verbatimModuleSyntax requires type-only imports.
import type { FormEvent } from 'react'

// Import the shared error-message narrowing types used when reporting add-contact failures.
import { ContactsApiError } from '../api/contactsClient'
// Import the server-side block client used by the more-options menu.
import { BlocksApiError, blockUser, listBlocks, unblockUser } from '../api/blocksClient'
// Import the metadata-only report client; message contents are never attached.
import { ReportsApiError, reportUser } from '../api/reportsClient'
// Import the session context so this screen can show identity and offer logout.
import { useAuth } from '../context/AuthContext'
// Import per-username theme helpers; tokens stay out of localStorage.
import { clearTheme } from '../theme'
import type { ThemePreference } from '../theme'
// Import the plaintext-export helpers; the UI always warns before writing to disk.
import { downloadTranscriptFile, formatTranscriptExport } from '../chat/exportTranscript'

// Import the extracted hooks this screen now composes instead of owning every effect
// and handler itself (Phase 3a of the pre-deployment review: ChatScreen had grown to
// ~790 lines covering contacts, theming, ML toggles, and the whole conversation state
// machine in one file; each concern now lives in its own testable hook module).
import { useAuthorizedRequest } from '../hooks/useAuthorizedRequest'
import { useChatTheme } from '../hooks/useChatTheme'
import { useContacts } from '../hooks/useContacts'
import { useEncryptedConversation } from '../hooks/useEncryptedConversation'
import { useScamClassifierPreference } from '../hooks/useScamClassifierPreference'
import { useUserSearch } from '../hooks/useUserSearch'
import type { ChatMessage } from '../hooks/useEncryptedConversation'

// Import the Phase 1 overlays this shell now mounts on demand.
import { ChatMoreMenu } from './ChatMoreMenu'
import { MessageActionsModal } from './MessageActionsModal'
import { Modal } from './Modal'
import { SettingsPanel } from './SettingsPanel'

// Name the mutually exclusive overlays so only one dialog is open at a time.
type ChatOverlay =
  | { kind: 'none' }
  | { kind: 'settings' }
  | { kind: 'more' }
  | { kind: 'message'; id: string }
  | { kind: 'export' }
  | { kind: 'report' }

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

// Decide whether a verified bubble matches the in-chat search query.
function messageMatchesQuery(message: ChatMessage, query: string): boolean {
  const trimmed = query.trim().toLowerCase()
  if (!trimmed) {
    return true
  }
  if (message.verificationFailed || message.plaintext === null) {
    return false
  }
  return message.plaintext.toLowerCase().includes(trimmed)
}

// Render the Slice 4+ encrypted conversation shell.
export function ChatScreen() {
  const { session, updateTokens, logout } = useAuth()

  // Hold the add-contact field until the owner saves a handle on the server.
  const [peerInput, setPeerInput] = useState('')
  // Hold a status string announced to assistive technology; shared by the contacts
  // form and the conversation state machine so only one line is ever shown at once.
  const [statusMessage, setStatusMessage] = useState(
    'Add a contact to start an encrypted chat.',
  )
  // Hold a form-level error distinct from per-message verification failures.
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  // Hold which overlay is open; 'none' means the chat shell has the focus.
  const [overlay, setOverlay] = useState<ChatOverlay>({ kind: 'none' })
  // Hold the in-chat search needle; empty means show the full transcript.
  const [inChatQuery, setInChatQuery] = useState('')
  // Hold whether the in-chat search field is visible under the header.
  const [searchOpen, setSearchOpen] = useState(false)
  // Hold the metadata-only report reason until the user submits it.
  const [reportReason, setReportReason] = useState('')
  // Hold whether the currently open peer is on this account's server-side block list.
  const [peerBlocked, setPeerBlocked] = useState(false)

  // Publish a rotated token pair into AuthContext; every wrapped call in the hooks
  // below reads the latest pair back out on its very next attempt. Memoized (like
  // AuthContext's own updateTokens) so effects that depend on it do not re-run every render.
  const handleTokensRotated = useCallback(
    (accessToken: string, refreshToken: string) => {
      updateTokens(accessToken, refreshToken)
    },
    [updateTokens],
  )

  // Wire the single-flight refresh-and-retry behavior once; contacts and the live
  // conversation both reuse it instead of each rolling their own.
  const authorizedRequest = useAuthorizedRequest(session, handleTokensRotated)

  const { contacts, addContact, removeContact } = useContacts(
    session?.username,
    authorizedRequest,
    setErrorMessage,
  )
  const { theme, themePreference, toggleTheme, setThemePreference } = useChatTheme(
    session?.username,
  )
  const classifier = useScamClassifierPreference()
  const userSearch = useUserSearch(peerInput, authorizedRequest, setErrorMessage)

  // Every hook must run unconditionally, on every render, per the Rules of Hooks —
  // including the one render where `session` flips back to null on logout, right
  // before AppRoutes swaps this screen back out for AuthScreen. useEncryptedConversation
  // tolerates a null session internally (see its own doc comment) precisely so this
  // call can sit before the early return below rather than after it.
  const {
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
    handleClearLocalTranscript,
    closeConversation,
  } = useEncryptedConversation(
    session,
    authorizedRequest,
    classifier.classify,
    setStatusMessage,
    setErrorMessage,
  )

  // This screen is only ever rendered while a session exists (see App.tsx routing);
  // this guard covers the one intermediate render described above.
  if (!session) {
    return null
  }
  const currentSession = session

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
      const saved = await addContact(peerUsername)
      setPeerInput('')
      setPeerBlocked(false)
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
    setPeerBlocked(false)
    void openConversation(username)
  }

  // Remove a handle from the address book; if that chat is open, tear it down locally.
  async function handleRemoveContact(username: string) {
    try {
      await removeContact(username)
      if (activeChat?.peerUsername === username) {
        closeConversation()
        setStatusMessage('Add a contact to start an encrypted chat.')
      }
    } catch (caught: unknown) {
      const message =
        caught instanceof ContactsApiError
          ? caught.message
          : 'Could not remove that contact. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  // Drop the per-user theme attribute, then clear the in-memory session.
  function handleLogout() {
    clearTheme()
    void logout()
  }

  // Persist an explicit theme preference from the Settings panel, including "system".
  function handleThemePreferenceChange(preference: ThemePreference) {
    setThemePreference(currentSession.username, preference)
  }

  // Ask the server whether the open peer is blocked so the menu can show Unblock.
  async function refreshPeerBlocked(username: string) {
    try {
      const rows = await authorizedRequest.request((accessToken) => listBlocks(accessToken))
      setPeerBlocked(rows.some((row) => row.username === username))
    } catch {
      // A block-list read failure must not block the rest of the menu.
      setPeerBlocked(false)
    }
  }

  async function handleBlockPeer() {
    if (!activeChat) {
      return
    }
    const username = activeChat.peerUsername
    try {
      await authorizedRequest.request((accessToken) => blockUser(accessToken, username))
      closeConversation()
      setPeerBlocked(true)
      setStatusMessage(`${username} is blocked. They cannot message you from this account.`)
    } catch (caught: unknown) {
      const message =
        caught instanceof BlocksApiError
          ? caught.message
          : 'Could not block that account. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  async function handleUnblockPeer() {
    if (!activeChat) {
      return
    }
    const username = activeChat.peerUsername
    try {
      await authorizedRequest.request((accessToken) => unblockUser(accessToken, username))
      setPeerBlocked(false)
      setStatusMessage(`${username} is unblocked.`)
    } catch (caught: unknown) {
      const message =
        caught instanceof BlocksApiError
          ? caught.message
          : 'Could not unblock that account. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  async function handleReportPeer() {
    if (!activeChat) {
      return
    }
    const reason = reportReason.trim()
    if (!reason) {
      setErrorMessage('Enter a reason for the report.')
      return
    }
    try {
      await authorizedRequest.request((accessToken) =>
        reportUser(accessToken, activeChat.peerUsername, reason),
      )
      setReportReason('')
      setOverlay({ kind: 'none' })
      setStatusMessage('Report filed. Message contents were not attached.')
    } catch (caught: unknown) {
      const message =
        caught instanceof ReportsApiError
          ? caught.message
          : 'Could not file that report. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  function handleConfirmExport() {
    if (!activeChat) {
      return
    }
    const contents = formatTranscriptExport(
      currentSession.username,
      activeChat.peerUsername,
      messages,
    )
    downloadTranscriptFile(`secure-chat-${activeChat.peerUsername}.txt`, contents)
    setOverlay({ kind: 'none' })
  }

  const visibleMessages = messages.filter((message) =>
    messageMatchesQuery(message, searchOpen ? inChatQuery : ''),
  )
  const selectedMessage =
    overlay.kind === 'message'
      ? messages.find((message) => message.id === overlay.id) ?? null
      : null

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
          {userSearch.results.length > 0 ? (
            <ul className="chat-search-results" aria-label="Matching accounts">
              {userSearch.results.map((hit) => (
                <li key={hit.username}>
                  <button
                    type="button"
                    className="chat-search-hit"
                    onClick={() => setPeerInput(hit.username)}
                  >
                    {hit.username}
                  </button>
                </li>
              ))}
            </ul>
          ) : peerInput.trim().length >= 2 && !userSearch.isSearching ? (
            <p className="chat-search-empty">No matching accounts.</p>
          ) : null}
          <button className="primary-button chat-add-button" type="submit" disabled={isConnecting}>
            {isConnecting ? 'Connecting…' : 'Add'}
          </button>
        </form>
        {contacts.length === 0 ? (
          <p className="chat-empty">No contacts yet. Add a username to start chatting.</p>
        ) : (
          <ul className="chat-contact-list">
            {contacts.map((contact) => (
              <li key={contact.id} className="chat-contact-row">
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
                <button
                  type="button"
                  className="text-button chat-contact-remove"
                  aria-label={`Remove ${contact.username}`}
                  onClick={() => void handleRemoveContact(contact.username)}
                >
                  Remove
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
                  checked={classifier.useDistilbert}
                  onChange={(event) => classifier.toggleDistilbert(event.target.checked)}
                />
                Use DistilBERT (large download)
              </label>
              <label className="chat-model-toggle">
                <input
                  type="checkbox"
                  checked={classifier.useLstm}
                  onChange={(event) => classifier.toggleLstm(event.target.checked)}
                />
                Use Word BiLSTM Best
              </label>
            </div>
            <button
              type="button"
              className="text-button"
              onClick={() => toggleTheme(currentSession.username)}
            >
              {theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            </button>
            <button
              type="button"
              className="text-button"
              onClick={() => setOverlay({ kind: 'settings' })}
            >
              Settings
            </button>
            <div className="chat-more-wrap">
              <button
                type="button"
                className="text-button"
                aria-haspopup="menu"
                aria-expanded={overlay.kind === 'more'}
                onClick={() => {
                  if (overlay.kind === 'more') {
                    setOverlay({ kind: 'none' })
                    return
                  }
                  if (activeChat) {
                    void refreshPeerBlocked(activeChat.peerUsername)
                  }
                  setOverlay({ kind: 'more' })
                }}
              >
                More
              </button>
              {overlay.kind === 'more' ? (
                <ChatMoreMenu
                  hasActiveChat={activeChat !== null}
                  peerBlocked={peerBlocked}
                  onSearch={() => {
                    setSearchOpen(true)
                    setOverlay({ kind: 'none' })
                  }}
                  onExport={() => setOverlay({ kind: 'export' })}
                  onClearLocal={() => {
                    handleClearLocalTranscript()
                    setStatusMessage('Local transcript cleared. The server copy is unchanged.')
                    setOverlay({ kind: 'none' })
                  }}
                  onBlock={() => {
                    setOverlay({ kind: 'none' })
                    void handleBlockPeer()
                  }}
                  onUnblock={() => {
                    setOverlay({ kind: 'none' })
                    void handleUnblockPeer()
                  }}
                  onReport={() => setOverlay({ kind: 'report' })}
                />
              ) : null}
            </div>
            <button type="button" className="text-button" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </header>

        {searchOpen ? (
          <div className="chat-inchat-search">
            <label htmlFor="in-chat-search">Search in chat</label>
            <input
              id="in-chat-search"
              type="search"
              value={inChatQuery}
              onChange={(event) => setInChatQuery(event.target.value)}
              placeholder="Filter decrypted messages on this device"
            />
            <button
              type="button"
              className="text-button"
              onClick={() => {
                setSearchOpen(false)
                setInChatQuery('')
              }}
            >
              Close search
            </button>
          </div>
        ) : null}

        <p className="chat-status" role="status">
          {statusMessage}
        </p>
        {errorMessage ? (
          <p className="auth-feedback auth-feedback-error" role="alert">
            {errorMessage}
          </p>
        ) : null}

        {classifier.classifierStatus ? (
          <p className="chat-status" role="status">
            {classifier.classifierStatus}
          </p>
        ) : null}

        {peerTyping && activeChat ? (
          <p className="chat-typing" role="status">
            {activeChat.peerUsername} is typing…
          </p>
        ) : null}

        <section className="chat-transcript" aria-label="Encrypted messages">
          {visibleMessages.length === 0 ? (
            <p className="chat-empty">
              {searchOpen && inChatQuery.trim()
                ? 'No messages match that search on this device.'
                : 'No messages yet. Ciphertext is relayed through the server; plaintext stays on this device.'}
            </p>
          ) : (
            <ul className="chat-message-list">
              {visibleMessages.map((message) => (
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
                  <button
                    type="button"
                    className="chat-bubble-button"
                    data-pending={message.pending ? 'true' : 'false'}
                    onClick={() => setOverlay({ kind: 'message', id: message.id })}
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
                        {message.revision > 0 ? (
                          <span className="chat-edited-flag"> (edited)</span>
                        ) : null}
                      </>
                    )}
                  </button>
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
      </main>

      {overlay.kind === 'settings' ? (
        <SettingsPanel
          themePreference={themePreference}
          onThemePreferenceChange={handleThemePreferenceChange}
          onClose={() => setOverlay({ kind: 'none' })}
          onLogout={handleLogout}
        />
      ) : null}

      {selectedMessage ? (
        <MessageActionsModal
          message={selectedMessage}
          onEdit={handleEditMessage}
          onDeleteForEveryone={handleDeleteMessage}
          onHideForMe={handleHideMessage}
          onClose={() => setOverlay({ kind: 'none' })}
        />
      ) : null}

      {overlay.kind === 'export' ? (
        <Modal title="Export chat" onClose={() => setOverlay({ kind: 'none' })}>
          <p>
            This download writes decrypted plaintext to a <code>.txt</code> file on this
            device. It is not end-to-end encrypted on disk.
          </p>
          <div className="chat-modal-actions">
            <button type="button" className="primary-button" onClick={handleConfirmExport}>
              Download plaintext
            </button>
            <button type="button" className="text-button" onClick={() => setOverlay({ kind: 'none' })}>
              Cancel
            </button>
          </div>
        </Modal>
      ) : null}

      {overlay.kind === 'report' ? (
        <Modal title="Report account" onClose={() => setOverlay({ kind: 'none' })}>
          <p>
            Reports are metadata only. Message contents are not attached and cannot be read
            by the server.
          </p>
          <form
            className="chat-report-form"
            onSubmit={(event) => {
              event.preventDefault()
              void handleReportPeer()
            }}
          >
            <label htmlFor="report-reason">Reason</label>
            <textarea
              id="report-reason"
              value={reportReason}
              onChange={(event) => setReportReason(event.target.value)}
              maxLength={500}
              rows={4}
              required
            />
            <div className="chat-modal-actions">
              <button type="submit" className="primary-button" disabled={!reportReason.trim()}>
                Submit report
              </button>
              <button
                type="button"
                className="text-button"
                onClick={() => setOverlay({ kind: 'none' })}
              >
                Cancel
              </button>
            </div>
          </form>
        </Modal>
      ) : null}
    </div>
  )
}
