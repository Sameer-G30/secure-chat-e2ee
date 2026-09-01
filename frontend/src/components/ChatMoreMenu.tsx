// Header "more options" menu ported from the legacy dropdown, with the known bugs
// fixed rather than copied: export warns that it writes plaintext to disk; report
// is metadata-only (contents are never attached); block is server-enforced;
// "clear chat" is local-only and says so. Each item only fires its action — the
// parent decides whether that action replaces this menu with another overlay
// (export/report) or just closes it (search/clear/block). Calling onClose after
// onExport would immediately dismiss the export warning dialog.

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
}: ChatMoreMenuProps) {
  return (
    <div className="chat-more-menu" role="menu" aria-label="More options">
      <button
        type="button"
        role="menuitem"
        className="chat-more-item"
        onClick={onSearch}
        disabled={!hasActiveChat}
      >
        Search in chat
      </button>
      <button
        type="button"
        role="menuitem"
        className="chat-more-item"
        onClick={onExport}
        disabled={!hasActiveChat}
      >
        Export chat
      </button>
      <button
        type="button"
        role="menuitem"
        className="chat-more-item"
        onClick={onClearLocal}
        disabled={!hasActiveChat}
      >
        Clear local transcript
      </button>
      {hasActiveChat && !peerBlocked ? (
        <button type="button" role="menuitem" className="chat-more-item" onClick={onBlock}>
          Block
        </button>
      ) : null}
      {hasActiveChat && peerBlocked ? (
        <button type="button" role="menuitem" className="chat-more-item" onClick={onUnblock}>
          Unblock
        </button>
      ) : null}
      {hasActiveChat ? (
        <button type="button" role="menuitem" className="chat-more-item" onClick={onReport}>
          Report
        </button>
      ) : null}
    </div>
  )
}
