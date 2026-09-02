// Import React's state/callback hooks used by the small amount of state this
// presentational shell still owns directly (shared status/error banners, the
// add-contact input, overlays, and conversation-start in-flight flag).
import { useCallback, useEffect, useRef, useState } from 'react'
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
import { formatMessageClock } from '../chat/messageClock'
import { previewTextForMessage, readAllLastMessagePreviews } from '../chat/lastMessagePreview'
import type { LastMessagePreview } from '../chat/lastMessagePreview'
import {
  clearConversationWallpaper,
  MAX_WALLPAPER_BYTES,
  readConversationWallpaper,
  writeConversationWallpaper,
} from '../chat/wallpaper'
import {
  fetchMyProfile,
  fetchUserProfile,
  patchMyProfile,
  uploadMyAvatar,
  UsersApiError,
} from '../api/usersClient'
import type { PublicProfile } from '../api/usersClient'

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
// Import the vrati icon set used on the sidebar, header, and circular send button.
import { IconGear, IconImage, IconMore, IconSearch, IconSend, IconVideo } from '../icons'
import { Avatar } from './Avatar'
import { ContactProfilePanel } from './ContactProfilePanel'

// Name the mutually exclusive overlays so only one dialog is open at a time.
type ChatOverlay =
  | { kind: 'none' }
  | { kind: 'settings' }
  | { kind: 'more' }
  | { kind: 'message'; id: string }
  | { kind: 'export' }
  | { kind: 'report' }
  | { kind: 'profile' }

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
  if (message.attachment) {
    return `${message.attachment.name} photo`.toLowerCase().includes(trimmed)
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
  // Hold the signed-in public profile used by the Settings form.
  const [myDisplayName, setMyDisplayName] = useState('')
  const [myBio, setMyBio] = useState('')
  const [myHasAvatar, setMyHasAvatar] = useState(false)
  // Hold last-message snippets keyed by peer username (IndexedDB, this device only).
  const [previews, setPreviews] = useState<Map<string, LastMessagePreview>>(new Map())
  // Hold the open peer's public profile for the contact-info panel.
  const [peerProfile, setPeerProfile] = useState<PublicProfile | null>(null)
  // Hold the wallpaper data URL for the open conversation.
  const [wallpaperUrl, setWallpaperUrl] = useState<string | null>(null)
  // Point at the hidden image-attach input so the paperclip button can open it.
  const imageInputRef = useRef<HTMLInputElement>(null)
  // Point at the hidden wallpaper input so Chat wallpaper can open it.
  const wallpaperInputRef = useRef<HTMLInputElement>(null)

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

  const { contacts, addContact, removeContact, clearUnread, reloadContacts } = useContacts(
    session?.username,
    authorizedRequest,
    setErrorMessage,
  )
  const { theme, themePreference, toggleTheme, setThemePreference } = useChatTheme(
    session?.username,
  )
  const classifier = useScamClassifierPreference(session?.username)
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
    handleSendImage,
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
    classifier.generation,
    classifier.scoringCheckpointId,
  )

  useEffect(() => {
    if (!session) {
      return
    }
    let cancelled = false
    void authorizedRequest
      .request((accessToken) => fetchMyProfile(accessToken))
      .then((profile) => {
        if (!cancelled) {
          setMyDisplayName(profile.displayName ?? '')
          setMyBio(profile.bio ?? '')
          setMyHasAvatar(profile.hasAvatar)
        }
      })
      .catch(() => {
        // Profile load is best-effort; Settings can still open with empty fields.
      })
    void readAllLastMessagePreviews(session.username).then((rows) => {
      if (!cancelled) {
        setPreviews(rows)
      }
    })
    return () => {
      cancelled = true
    }
  }, [session?.username, authorizedRequest.request])

  useEffect(() => {
    if (!session || !activeChat) {
      setWallpaperUrl(null)
      setPeerProfile(null)
      return
    }
    setWallpaperUrl(readConversationWallpaper(session.username, activeChat.conversationId))
    let cancelled = false
    void authorizedRequest
      .request((accessToken) => fetchUserProfile(accessToken, activeChat.peerUsername))
      .then((profile) => {
        if (!cancelled) {
          setPeerProfile(profile)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPeerProfile(null)
        }
      })
    return () => {
      cancelled = true
    }
  }, [session?.username, activeChat?.conversationId, activeChat?.peerUsername, authorizedRequest.request])

  useEffect(() => {
    if (!activeChat) {
      return
    }
    const last = [...messages].reverse().find(
      (row) => !row.verificationFailed && row.plaintext !== null,
    )
    if (!last || last.plaintext === null) {
      return
    }
    const snippet = previewTextForMessage(
      last.plaintext,
      last.direction,
      last.attachment !== null,
    )
    setPreviews((existing) => {
      const next = new Map(existing)
      next.set(activeChat.peerUsername, snippet)
      return next
    })
  }, [messages, activeChat?.peerUsername])

  // Reload unread badges when this tab is focused or becomes visible again.
  useEffect(() => {
    function onVisible() {
      // Skip the hidden half of visibilitychange; focus events are always "visible".
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return
      }
      void reloadContacts().catch(() => {
        // Unread refresh is best-effort.
      })
    }
    window.addEventListener('focus', onVisible)
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      window.removeEventListener('focus', onVisible)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [reloadContacts])

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
      clearUnread(saved.username)
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
    clearUnread(username)
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

  async function handleSaveProfile(displayName: string, bio: string) {
    try {
      const saved = await authorizedRequest.request((accessToken) =>
        patchMyProfile(accessToken, { displayName, bio }),
      )
      setMyDisplayName(saved.displayName ?? '')
      setMyBio(saved.bio ?? '')
      setMyHasAvatar(saved.hasAvatar)
    } catch (caught: unknown) {
      const message =
        caught instanceof UsersApiError
          ? caught.message
          : 'Could not save your profile. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  async function handleUploadAvatar(file: File) {
    try {
      const saved = await authorizedRequest.request((accessToken) =>
        uploadMyAvatar(accessToken, file),
      )
      setMyDisplayName(saved.displayName ?? myDisplayName)
      setMyBio(saved.bio ?? myBio)
      setMyHasAvatar(saved.hasAvatar)
    } catch (caught: unknown) {
      const message =
        caught instanceof UsersApiError
          ? caught.message
          : 'Could not upload that photo. Please try again shortly.'
      setErrorMessage(message)
    }
  }

  function handleWallpaperFile(file: File) {
    if (!activeChat) {
      return
    }
    if (file.size > MAX_WALLPAPER_BYTES) {
      setErrorMessage('That wallpaper is too large for this device.')
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string' && activeChat) {
        writeConversationWallpaper(
          currentSession.username,
          activeChat.conversationId,
          reader.result,
        )
        setWallpaperUrl(reader.result)
      }
    }
    reader.readAsDataURL(file)
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

  // Count in-chat search hits so the header chip can show how many rows remain.
  const searchMatchCount = visibleMessages.length
  // Label sent bubbles with Sending until the server ack, then Sent.
  function pendingLabel(message: ChatMessage): string {
    return formatMessageClock(message.createdAt)
  }

  function deliveryTickClass(message: ChatMessage): string {
    if (message.direction !== 'sent') {
      return ''
    }
    if (message.pending || message.deliveryStatus === 'pending') {
      return ''
    }
    if (message.deliveryStatus === 'read') {
      return 'status-read'
    }
    if (message.deliveryStatus === 'delivered') {
      return 'status-delivered'
    }
    return 'status-sent'
  }

  function deliveryTick(message: ChatMessage): string {
    if (message.direction !== 'sent') {
      return ''
    }
    if (message.pending || message.deliveryStatus === 'pending') {
      return ''
    }
    if (message.deliveryStatus === 'delivered' || message.deliveryStatus === 'read') {
      return '✓✓'
    }
    return '✓'
  }

  return (
    <div className="chat-app">
      <aside className="sidebar" aria-label="Contacts">
        <div className="sidebar-header">
          <Avatar
            username={currentSession.username}
            accessToken={currentSession.accessToken}
            className="user-avatar"
            initials={initialsFromUsername(currentSession.username)}
          />
          <div className="user-info">
            <div className="user-name">{myDisplayName.trim() || currentSession.username}</div>
            <div className="user-status">Online</div>
          </div>
          <div className="sidebar-actions">
            <button
              type="button"
              className="icon-btn"
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              onClick={() => toggleTheme(currentSession.username)}
            >
              {theme === 'dark' ? '☀' : '☾'}
            </button>
            <button
              type="button"
              className="icon-btn"
              aria-label="Settings"
              onClick={() => setOverlay({ kind: 'settings' })}
            >
              <IconGear />
            </button>
          </div>
        </div>

        <form className="search-bar" onSubmit={(event) => void handleAddContact(event)}>
          <div className="search-input-wrapper">
            <IconSearch size={18} />
            <label className="visually-hidden" htmlFor="add-contact">
              Add contact
            </label>
            <input
              id="add-contact"
              name="addContact"
              type="text"
              autoComplete="username"
              placeholder="Search users..."
              value={peerInput}
              onChange={(event) => setPeerInput(event.target.value)}
              required
            />
          </div>
          <button className="search-btn" type="submit" disabled={isConnecting}>
            {isConnecting ? '…' : 'Add'}
          </button>
        </form>

        {userSearch.results.length > 0 ? (
          <div className="search-results">
            <div className="search-results-header">Matching accounts</div>
            <ul className="search-results-list" aria-label="Matching accounts">
              {userSearch.results.map((hit) => (
                <li key={hit.username}>
                  <button
                    type="button"
                    className="search-result-item"
                    aria-label={hit.username}
                    onClick={() => setPeerInput(hit.username)}
                  >
                    <span className="search-result-info">
                      <span className="search-result-avatar" aria-hidden="true">
                        {initialsFromUsername(hit.username)}
                      </span>
                      <span className="search-result-name">{hit.username}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : peerInput.trim().length >= 2 && !userSearch.isSearching ? (
          <p className="chat-search-empty">No matching accounts.</p>
        ) : null}

        {contacts.length === 0 ? (
          <div className="empty-state">No contacts yet. Add a username to start chatting.</div>
        ) : (
          <ul className="contacts-list">
            {contacts.map((contact) => (
              <li key={contact.id} className="contact-row">
                <button
                  type="button"
                  className={activeChat?.peerId === contact.id ? 'contact-item active' : 'contact-item'}
                  aria-label={contact.username}
                  onClick={() => handleSelectContact(contact.username)}
                  disabled={isConnecting}
                >
                  <Avatar
                    username={contact.username}
                    accessToken={currentSession.accessToken}
                    className="contact-avatar"
                    initials={initialsFromUsername(contact.username)}
                  />
                  <span className="contact-info">
                    <span className="contact-name">
                      {contact.displayName?.trim() || contact.username}
                    </span>
                    <span className="contact-preview">
                      {previews.get(contact.username)?.preview ??
                        (activeChat?.peerId === contact.id
                          ? peerOnline
                            ? 'Online'
                            : 'Offline'
                          : 'Tap to chat')}
                    </span>
                  </span>
                  {contact.unreadCount > 0 ? (
                    <span className="unread-badge">{contact.unreadCount}</span>
                  ) : null}
                </button>
                <button
                  type="button"
                  className="contact-remove"
                  aria-label={`Remove ${contact.username}`}
                  onClick={() => void handleRemoveContact(contact.username)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="sidebar-footer">
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
        </div>
      </aside>

      <main
        className={wallpaperUrl ? 'chat-area has-wallpaper' : 'chat-area'}
        style={wallpaperUrl ? { backgroundImage: `url(${wallpaperUrl})` } : undefined}
      >
        <h1 className="chat-product-heading">Secure Chat</h1>
        <header className="chat-header">
          <div className="chat-header-left">
            {activeChat ? (
              <>
                <button
                  type="button"
                  className="chat-header-identity"
                  onClick={() => setOverlay({ kind: 'profile' })}
                >
                  <Avatar
                    username={activeChat.peerUsername}
                    accessToken={currentSession.accessToken}
                    className="chat-contact-avatar"
                    initials={initialsFromUsername(activeChat.peerUsername)}
                  />
                  <div>
                    <div className="chat-contact-name">
                      {peerProfile?.displayName?.trim() || activeChat.peerUsername}
                    </div>
                    <div className="chat-contact-status">
                      {peerBlocked
                        ? 'Blocked'
                        : peerOnline
                          ? 'Online · end-to-end encrypted'
                          : 'End-to-end encrypted'}
                    </div>
                  </div>
                </button>
              </>
            ) : (
              <div>
                <div className="chat-contact-name">Select a contact</div>
                <div className="chat-contact-status">End-to-end encrypted</div>
              </div>
            )}
          </div>
          <div className="chat-header-actions">
            <button
              type="button"
              className="icon-btn"
              aria-label="Search messages"
              onClick={() => setSearchOpen(true)}
              disabled={!activeChat}
            >
              <IconSearch />
            </button>
            <button
              type="button"
              className="icon-btn"
              aria-label="More"
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
              <IconMore />
            </button>
          </div>
        </header>

        {overlay.kind === 'more' ? (
          <ChatMoreMenu
            hasActiveChat={activeChat !== null}
            peerBlocked={peerBlocked}
            onDismiss={() => setOverlay({ kind: 'none' })}
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
            onContactInfo={() => setOverlay({ kind: 'profile' })}
            onWallpaper={() => wallpaperInputRef.current?.click()}
            onClearWallpaper={() => {
              if (activeChat) {
                clearConversationWallpaper(currentSession.username, activeChat.conversationId)
                setWallpaperUrl(null)
              }
              setOverlay({ kind: 'none' })
            }}
          />
        ) : null}

        {searchOpen ? (
          <div className="search-bar-chat">
            <div className="search-bar-chat-input">
              <IconSearch size={16} />
              <label className="visually-hidden" htmlFor="in-chat-search">
                Search in chat
              </label>
              <input
                id="in-chat-search"
                type="search"
                value={inChatQuery}
                onChange={(event) => setInChatQuery(event.target.value)}
                placeholder="Filter decrypted messages on this device"
              />
            </div>
            <div className="search-bar-chat-actions">
              <span className="search-results-count">
                {inChatQuery.trim() ? `${searchMatchCount}` : ''}
              </span>
              <button
                type="button"
                className="search-close-btn"
                onClick={() => {
                  setSearchOpen(false)
                  setInChatQuery('')
                }}
              >
                Close search
              </button>
            </div>
          </div>
        ) : null}

        {peerBlocked && activeChat ? (
          <div className="blocked-banner">
            <div className="blocked-banner-content">
              <span className="blocked-icon">🔒</span>
              <div>
                <div className="blocked-title">You have blocked {activeChat.peerUsername}</div>
                <div className="blocked-subtitle">They cannot message you from this account.</div>
              </div>
              <button type="button" className="unblock-btn" onClick={() => void handleUnblockPeer()}>
                Unblock
              </button>
            </div>
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

        {!activeChat ? (
          <div className="empty-chat-state">
            <div className="empty-chat-icon" aria-hidden="true">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
              </svg>
            </div>
            <h3>Select a contact</h3>
            <p>Choose someone from your contacts to start chatting</p>
          </div>
        ) : visibleMessages.length === 0 ? (
          <div className="messages">
            <p className="empty-chat">
              {searchOpen && inChatQuery.trim()
                ? 'No messages match that search on this device.'
                : 'No messages yet. Ciphertext is relayed through the server; plaintext stays on this device.'}
            </p>
          </div>
        ) : (
          <ul className="messages" aria-label="Encrypted messages">
            {visibleMessages.map((message) => {
                  const bubbleClass = message.verificationFailed
                    ? 'message failed'
                    : message.direction === 'sent'
                      ? 'message sent'
                      : 'message received'
                  const accessibleName = message.verificationFailed
                    ? undefined
                    : message.attachment
                      ? message.attachment.name
                      : (message.plaintext ?? undefined)
                  const tickClass = deliveryTickClass(message)
                  const tick = deliveryTick(message)
                  return (
                    <li key={message.id} className="message-wrapper">
                      <button
                        type="button"
                        className={bubbleClass}
                        data-pending={message.pending ? 'true' : 'false'}
                        aria-label={accessibleName}
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
                            {message.attachment ? (
                              <span className="message-attachment">
                                {message.attachment.objectUrl ? (
                                  <img
                                    src={message.attachment.objectUrl}
                                    alt=""
                                    className="message-image"
                                  />
                                ) : (
                                  <span className="message-image-placeholder">
                                    {message.attachment.failed
                                      ? 'Photo failed to load'
                                      : 'Loading photo…'}
                                  </span>
                                )}
                              </span>
                            ) : (
                              <span className="message-text">
                                {message.plaintext}
                                {message.revision > 0 ? (
                                  <span className="edited-indicator"> (edited)</span>
                                ) : null}
                              </span>
                            )}
                            <span className="message-time">
                              {pendingLabel(message)}
                              {tick ? (
                                <span className={tickClass} aria-hidden="true">
                                  {tick}
                                </span>
                              ) : null}
                            </span>
                          </>
                        )}
                      </button>
                    </li>
                  )
                })}
          </ul>
        )}

        {peerBlocked && activeChat ? (
          <div className="blocked-input-banner">
            <span>You can&apos;t send messages to a blocked user.</span>
            <button type="button" className="unblock-link" onClick={() => void handleUnblockPeer()}>
              Unblock to chat
            </button>
          </div>
        ) : (
          <form className="input-area" onSubmit={handleSend}>
            <input
              ref={imageInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,image/gif"
              className="visually-hidden"
              aria-hidden="true"
              tabIndex={-1}
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) {
                  void handleSendImage(file)
                }
                event.target.value = ''
              }}
            />
            <button
              type="button"
              className="input-attach"
              aria-label="Attach image"
              disabled={!activeChat}
              onClick={() => imageInputRef.current?.click()}
            >
              <IconImage />
            </button>
            {/* Video stays a disabled placeholder; sealed video blobs are not implemented yet. */}
            <button
              type="button"
              className="input-attach"
              aria-label="Video attachments coming later"
              title="Video attachments coming later"
              disabled
            >
              <IconVideo />
            </button>
            <label className="visually-hidden" htmlFor="chat-draft">
              Message
            </label>
            <input
              id="chat-draft"
              name="draft"
              type="text"
              autoComplete="off"
              placeholder="Type a message..."
              value={draft}
              onChange={(event) => handleDraftChange(event.target.value)}
              disabled={!activeChat}
            />
            <button
              className="btn-send"
              type="submit"
              aria-label="Send"
              disabled={!activeChat || !draft.trim()}
            >
              <IconSend />
            </button>
          </form>
        )}
      </main>

      {overlay.kind === 'settings' ? (
        <SettingsPanel
          themePreference={themePreference}
          onThemePreferenceChange={handleThemePreferenceChange}
          onClose={() => setOverlay({ kind: 'none' })}
          onLogout={handleLogout}
          profileUsername={currentSession.username}
          profileAccessToken={currentSession.accessToken}
          profileDisplayName={myDisplayName}
          profileBio={myBio}
          profileHasAvatar={myHasAvatar}
          onSaveProfile={handleSaveProfile}
          onUploadAvatar={handleUploadAvatar}
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

      {overlay.kind === 'profile' && activeChat ? (
        <ContactProfilePanel
          username={activeChat.peerUsername}
          accessToken={currentSession.accessToken}
          profile={peerProfile}
          online={peerOnline}
          blocked={peerBlocked}
          onClose={() => setOverlay({ kind: 'none' })}
          onSearch={() => {
            setSearchOpen(true)
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

      <input
        ref={wallpaperInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="visually-hidden"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) {
            handleWallpaperFile(file)
          }
          event.target.value = ''
          setOverlay({ kind: 'none' })
        }}
      />
    </div>
  )
}
