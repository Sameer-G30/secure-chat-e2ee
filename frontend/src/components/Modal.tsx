// Reusable overlay used by settings, report, search, export-warning, and message-action
// confirmations. Ported from the legacy Modal without Firebase, and without
// assigning untrusted strings into the DOM as HTML — title and children are
// React text nodes only.

// Import the ReactNode type separately; verbatimModuleSyntax requires type-only imports.
import type { ReactNode } from 'react'

// Describe the small surface this overlay exposes.
export interface ModalProps {
  // Visible heading announced to assistive technology.
  title: string
  // Body of the dialog (forms, copy, buttons).
  children: ReactNode
  // Close the overlay (Escape, backdrop click, or an explicit Cancel).
  onClose: () => void
}

// Render a labelled dialog over the chat shell.
export function Modal({ title, children, onClose }: ModalProps) {
  return (
    <div
      className="chat-modal-backdrop"
      role="presentation"
      onClick={(event) => {
        // Only the dimmed backdrop dismisses; clicks inside the panel stay put.
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
      <div className="chat-modal" role="dialog" aria-modal="true" aria-labelledby="chat-modal-title">
        <header className="chat-modal-header">
          <h2 id="chat-modal-title">{title}</h2>
          <button type="button" className="text-button" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="chat-modal-body">{children}</div>
      </div>
    </div>
  )
}
