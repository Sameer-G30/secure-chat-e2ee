// Per-bubble actions with vrati message-actions chrome. Edit is sender-only,
// v2-only, and refused while the row is still pending; delete-for-everyone is
// sender-only; hide-for-me is available on any already-accepted row.

// Import React's state hook used by the inline edit field.
import { useState } from 'react'
// Import the in-memory bubble shape this modal inspects.
import type { ChatMessage } from '../hooks/useEncryptedConversation'
// Import the dismiss glyph used on the modal header.
import { IconClose } from '../icons'

// Describe the handlers ChatScreen wires into this modal.
export interface MessageActionsModalProps {
  // Carry the bubble the user clicked.
  message: ChatMessage
  // Re-encrypt and resend an own v2 message; no-op guards live in the hook.
  onEdit: (id: string, plaintext: string) => Promise<void>
  // Hard-delete an own message for every participant.
  onDeleteForEveryone: (id: string) => Promise<void>
  // Hide this row from this account's own future history only.
  onHideForMe: (id: string) => Promise<void>
  // Close the overlay.
  onClose: () => void
}

// Render edit / delete / hide / copy for one selected bubble.
export function MessageActionsModal({
  message,
  onEdit,
  onDeleteForEveryone,
  onHideForMe,
  onClose,
}: MessageActionsModalProps) {
  // Hold the in-progress edit text, starting from the verified plaintext.
  const [editDraft, setEditDraft] = useState(message.plaintext ?? '')
  // Hold whether the inline editor is open (own, v2, not pending only).
  const [isEditing, setIsEditing] = useState(false)

  // Own, already-accepted, v2 messages can be edited.
  const canEdit =
    message.direction === 'sent' &&
    !message.pending &&
    !message.verificationFailed &&
    message.clientMessageId !== null
  // Own, already-accepted messages can be hard-deleted for everyone.
  const canDeleteForEveryone = message.direction === 'sent' && !message.pending
  // Any already-accepted row can be hidden from this account's own history.
  const canHide = !message.pending
  // Copy is only meaningful for verified plaintext.
  const canCopy = !message.verificationFailed && message.plaintext !== null

  async function handleSaveEdit() {
    const trimmed = editDraft.trim()
    if (!trimmed) {
      return
    }
    await onEdit(message.id, trimmed)
    onClose()
  }

  async function handleCopy() {
    if (message.plaintext === null) {
      return
    }
    try {
      await navigator.clipboard.writeText(message.plaintext)
    } catch {
      // A locked-down test environment or missing clipboard permission is not fatal.
    }
    onClose()
  }

  return (
    <div
      className="message-actions-overlay"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose()
        }
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          onClose()
        }
      }}
    >
      <div className="message-actions-modal" role="dialog" aria-modal="true" aria-labelledby="message-actions-title">
        <div className="message-actions-header">
          <span id="message-actions-title">Message actions</span>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
            <IconClose size={20} />
          </button>
        </div>
        <div className="message-actions-content">
          {isEditing ? (
            <form
              className="chat-edit-form"
              onSubmit={(event) => {
                event.preventDefault()
                void handleSaveEdit()
              }}
            >
              <label htmlFor="chat-edit-draft">Edited message</label>
              <textarea
                id="chat-edit-draft"
                value={editDraft}
                onChange={(event) => setEditDraft(event.target.value)}
                rows={3}
              />
              <div className="chat-modal-actions">
                <button type="submit" className="primary-button" disabled={!editDraft.trim()}>
                  Save edit
                </button>
                <button type="button" className="text-button" onClick={() => setIsEditing(false)}>
                  Cancel
                </button>
              </div>
            </form>
          ) : (
            <>
              {canCopy ? (
                <button type="button" className="message-action-btn" onClick={() => void handleCopy()}>
                  Copy
                </button>
              ) : null}
              {canEdit ? (
                <button type="button" className="message-action-btn" onClick={() => setIsEditing(true)}>
                  Edit
                </button>
              ) : null}
              {canDeleteForEveryone ? (
                <button
                  type="button"
                  className="message-action-btn"
                  onClick={() => void onDeleteForEveryone(message.id).then(onClose)}
                >
                  Delete for everyone
                </button>
              ) : null}
              {canHide ? (
                <button
                  type="button"
                  className="message-action-btn"
                  onClick={() => void onHideForMe(message.id).then(onClose)}
                >
                  Hide for me
                </button>
              ) : null}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
