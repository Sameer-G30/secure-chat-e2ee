// Header "more options" menu with vrati dropdown chrome. Export warns that it
// writes plaintext to disk; report is metadata-only; block is server-enforced;
// "clear chat" is local-only and says so. Each item only fires its action.

// Describe the handlers ChatScreen wires into this menu.
export interface ChatMoreMenuProps {
  // True when a conversation is open, which unlocks per-peer actions.
  hasActiveChat: boolean
  // True when the open peer is already on this account's server-side block list.
  peerBlocked: boolean
  // Run in-chat search over decrypted in-memory text.
  onSearch: () => void
  // Open the export-warning confirmation before writing a .txt file.
  onExport: () => void
  // Drop this tab's in-memory transcript without touching the server.
  onClearLocal: () => void
  // Block the open peer server-side, then close the conversation.
  onBlock: () => void
  // Remove the open peer from this account's block list.
  onUnblock: () => void
  // Open the metadata-only report form.
  onReport: () => void
  // Open the wallpaper file picker for this conversation.
  onWallpaper: () => void
  // Remove the local wallpaper for this conversation.
  onClearWallpaper: () => void
  // Open the contact profile slide-over.
  onContactInfo: () => void
  // Dismiss the overlay when the dimmed backdrop is clicked.
  onDismiss: () => void
}

// Render the overflow actions for the open conversation (or a reduced set if none).
export function ChatMoreMenu({
  hasActiveChat,
  peerBlocked,
  onSearch,
  onExport,
  onClearLocal,
  onBlock,
  onUnblock,
  onReport,
  onWallpaper,
  onClearWallpaper,
  onContactInfo,
  onDismiss,
}: ChatMoreMenuProps) {
  return (
    <div
      className="dropdown-overlay"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onDismiss()
        }
      }}
    >
      <div className="dropdown-menu" role="menu" aria-label="More options">
        <div className="dropdown-header">
          <span>Chat Options</span>
        </div>
        <div className="dropdown-items">
          <button
            type="button"
            role="menuitem"
            className="dropdown-item"
            onClick={onSearch}
            disabled={!hasActiveChat}
          >
            Search in chat
          </button>
          <button
            type="button"
            role="menuitem"
            className="dropdown-item"
            onClick={onExport}
            disabled={!hasActiveChat}
          >
            Export chat
          </button>
          <button
            type="button"
            role="menuitem"
            className="dropdown-item"
            onClick={onClearLocal}
            disabled={!hasActiveChat}
          >
            Clear local transcript
          </button>
          <button
            type="button"
            role="menuitem"
            className="dropdown-item"
            onClick={onContactInfo}
            disabled={!hasActiveChat}
          >
            Contact info
          </button>
          <button
            type="button"
            role="menuitem"
            className="dropdown-item"
            onClick={onWallpaper}
            disabled={!hasActiveChat}
          >
            Chat wallpaper
          </button>
          <button
            type="button"
            role="menuitem"
            className="dropdown-item"
            onClick={onClearWallpaper}
            disabled={!hasActiveChat}
          >
            Remove wallpaper
          </button>
          {hasActiveChat && !peerBlocked ? (
            <button type="button" role="menuitem" className="dropdown-item dropdown-item-danger" onClick={onBlock}>
              Block
            </button>
          ) : null}
          {hasActiveChat && peerBlocked ? (
            <button type="button" role="menuitem" className="dropdown-item" onClick={onUnblock}>
              Unblock
            </button>
          ) : null}
          {hasActiveChat ? (
            <button type="button" role="menuitem" className="dropdown-item dropdown-item-danger" onClick={onReport}>
              Report
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}
