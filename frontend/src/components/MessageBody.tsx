// Render verified message text with real web anchors.
// https, http, www, and bare hosts such as youtube.com are links. Other schemes stay text.
// Nothing is assigned as HTML.

// Import the splitter. Bare hosts get an https href. javascript: and data: stay text.
import { splitHttpLinks } from '../chat/messageLinks'

// Props for one verified text bubble that contains at least one http(s) URL.
export interface MessageBodyProps {
  // Verified plaintext. Rendered as text nodes, never as HTML.
  text: string
  // True when this bubble is showing the scam banner.
  flagged: boolean
  // Ask the chat shell to confirm before opening a link in a flagged bubble.
  onFlaggedLink: (href: string) => void
}

// Render the bubble body. Unflagged links navigate immediately. Flagged links do not.
export function MessageBody({ text, flagged, onFlaggedLink }: MessageBodyProps) {
  // Split into text and http(s) links. javascript: and data: stay in text runs.
  const parts = splitHttpLinks(text)
  // React text nodes and anchors, in order.
  return (
    <>
      {parts.map((part, index) =>
        part.kind === 'link' && part.href ? (
          <a
            key={index}
            className="message-link"
            href={part.href}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(event) => {
              // The surrounding words open message actions. This click is only the link.
              event.stopPropagation()
              // A flagged bubble must not navigate until the sender confirms.
              if (flagged) {
                // Cancel the browser's follow of href.
                event.preventDefault()
                // The shell shows Continue / Cancel and names this URL.
                onFlaggedLink(part.href ?? '')
              }
            }}
            onAuxClick={(event) => {
              // Middle-click would follow href without the primary click handler.
              if (flagged) {
                // Keep a flagged link inside the chat until Continue.
                event.preventDefault()
                // Do not open message actions from a middle-click on the link either.
                event.stopPropagation()
              }
            }}
          >
            {part.text}
          </a>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </>
  )
}
