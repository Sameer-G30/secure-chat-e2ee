// Per-bubble actions ported from the legacy message-actions modal, with the known
// bugs fixed rather than copied: edit is sender-only, v2-only, and refused while
// the row is still pending an "accepted" ack; delete-for-everyone is sender-only;
// hide-for-me is available on any already-accepted row; copy never writes
// verification-failed text to the clipboard.

// Import React's state hook used by the inline edit field.
import { useState } from 'react'

// Import the reusable overlay this panel renders inside.
import { Modal } from './Modal'
// Import the in-memory bubble shape this modal inspects.
import type { ChatMessage } from '../hooks/useEncryptedConversation'

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

  // Own, already-accepted, v2 messages can be edited; history from before editing
  // existed has a null clientMessageId and cannot be bound to a new revision.
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
    <Modal title="Message actions" onClose={onClose}>
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
        <div className="chat-modal-actions chat-modal-actions-stack">
          {canCopy ? (
            <button type="button" className="text-button" onClick={() => void handleCopy()}>
              Copy
            </button>
          ) : null}
          {canEdit ? (
            <button type="button" className="text-button" onClick={() => setIsEditing(true)}>
              Edit
            </button>
          ) : null}
          {canDeleteForEveryone ? (
            <button
              type="button"
              className="text-button"
              onClick={() => void onDeleteForEveryone(message.id).then(onClose)}
            >
              Delete for everyone
            </button>
          ) : null}
          {canHide ? (
            <button
              type="button"
              className="text-button"
              onClick={() => void onHideForMe(message.id).then(onClose)}
            >
              Hide for me
            </button>
          ) : null}
        </div>
      )}
    </Modal>
  )
}
