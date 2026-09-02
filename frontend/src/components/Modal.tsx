// Reusable overlay used by report, export-warning, and other confirmations.
// Chrome matches frontend/vrati `.modal-overlay` / `.modal-content`. Title and
// children are React text nodes only — untrusted strings are never assigned as HTML.

// Import the ReactNode type separately; verbatimModuleSyntax requires type-only imports.
import type { ReactNode } from 'react'
// Import the dismiss glyph used on the modal header.
import { IconClose } from '../icons'

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
      className="modal-overlay"
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
      <div className="modal-content" role="dialog" aria-modal="true" aria-labelledby="chat-modal-title">
        <header className="modal-header">
          <h3 id="chat-modal-title">{title}</h3>
          <button type="button" className="modal-close" aria-label="Close" onClick={onClose}>
            <IconClose size={20} />
          </button>
        </header>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  )
}
