// Full-screen photo viewer. Zoom stays between 1× and 4×. Avatars omit Forward.

// Import React hooks used for focus, zoom, and pan.
import { useEffect, useRef, useState } from 'react'
// Import the close glyph.
import { IconClose } from '../icons'

// Clamp the lightbox scale.
const MIN_SCALE = 1
const MAX_SCALE = 4

// Describe the photo and the optional forward action.
export interface ImageLightboxProps {
  // Object URL of a decrypted chat image or a public avatar.
  src: string
  // Accessible name. A filename or a username. Rendered as text, never HTML.
  title: string
  // Chat images can be forwarded. Public avatars cannot.
  allowForward: boolean
  // Close on Escape, the backdrop, or the close button.
  onClose: () => void
  // Open the contact picker. Omitted when allowForward is false.
  onForward?: () => void
}

// Keep keyboard focus inside the dialog until it closes.
function trapTab(event: KeyboardEvent, root: HTMLElement) {
  // Only Tab is trapped. Escape is handled by the caller.
  if (event.key !== 'Tab') {
    return
  }
  // Every control inside the dialog.
  const focusable = Array.from(
    root.querySelectorAll<HTMLElement>('button, [href], input, [tabindex]:not([tabindex="-1"])'),
  ).filter((node) => !node.hasAttribute('disabled'))
  // Nothing to cycle.
  if (focusable.length === 0) {
    event.preventDefault()
    return
  }
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  const active = document.activeElement
  // Shift+Tab on the first control wraps to the last.
  if (event.shiftKey && active === first) {
    event.preventDefault()
    last?.focus()
    return
  }
  // Tab on the last control wraps to the first.
  if (!event.shiftKey && active === last) {
    event.preventDefault()
    first?.focus()
  }
}

// Render one photo with zoom, pan, and an optional forward action.
export function ImageLightbox({
  src,
  title,
  allowForward,
  onClose,
  onForward,
}: ImageLightboxProps) {
  // Current zoom. 1 is the fitted photo.
  const [scale, setScale] = useState(MIN_SCALE)
  // Pan offset in pixels, used only while zoomed.
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  // The dialog node used for the focus trap.
  const dialogRef = useRef<HTMLDivElement>(null)
  // True while a pointer is dragging the zoomed photo.
  const dragRef = useRef<{ x: number; y: number; originX: number; originY: number } | null>(null)
  // Pinch start distance, when two touches are down.
  const pinchRef = useRef<number | null>(null)

  // Focus the dialog and restore focus when it closes.
  useEffect(() => {
    const root = dialogRef.current
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
    root?.focus()
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose()
        return
      }
      if (root) {
        trapTab(event, root)
      }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      previouslyFocused?.focus()
    }
  }, [onClose])

  // Wheel zoom. The listener is non-passive so it can prevent the page from scrolling.
  useEffect(() => {
    const root = dialogRef.current
    if (!root) {
      return
    }
    function onWheel(event: WheelEvent) {
      event.preventDefault()
      setScale((current) => {
        const next = current + (event.deltaY < 0 ? 0.25 : -0.25)
        return Math.min(MAX_SCALE, Math.max(MIN_SCALE, next))
      })
    }
    root.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      root.removeEventListener('wheel', onWheel)
    }
  }, [])

  // Reset pan when the user returns to 1×.
  useEffect(() => {
    if (scale === MIN_SCALE) {
      setOffset({ x: 0, y: 0 })
    }
  }, [scale])

  function zoomBy(delta: number) {
    setScale((current) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, current + delta)))
  }

  return (
    <div
      className="lightbox-backdrop"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose()
        }
      }}
    >
      <div
        ref={dialogRef}
        className="lightbox-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
      >
        <div className="lightbox-toolbar">
          <button type="button" onClick={() => zoomBy(0.5)}>
            Zoom in
          </button>
          <button type="button" onClick={() => zoomBy(-0.5)}>
            Zoom out
          </button>
          <button
            type="button"
            onClick={() => {
              setScale(MIN_SCALE)
              setOffset({ x: 0, y: 0 })
            }}
          >
            Reset
          </button>
          {allowForward && onForward ? (
            <button type="button" onClick={onForward}>
              Forward
            </button>
          ) : null}
          <button type="button" aria-label="Close photo" onClick={onClose}>
            <IconClose size={20} />
          </button>
        </div>
        <div
          className="lightbox-stage"
          onPointerDown={(event) => {
            if (scale <= MIN_SCALE) {
              return
            }
            dragRef.current = {
              x: event.clientX,
              y: event.clientY,
              originX: offset.x,
              originY: offset.y,
            }
          }}
          onPointerMove={(event) => {
            const drag = dragRef.current
            if (!drag) {
              return
            }
            setOffset({
              x: drag.originX + (event.clientX - drag.x),
              y: drag.originY + (event.clientY - drag.y),
            })
          }}
          onPointerUp={() => {
            dragRef.current = null
          }}
          onTouchStart={(event) => {
            if (event.touches.length === 2) {
              const [a, b] = [event.touches[0], event.touches[1]]
              if (a && b) {
                const dx = a.clientX - b.clientX
                const dy = a.clientY - b.clientY
                pinchRef.current = Math.hypot(dx, dy)
              }
            }
          }}
          onTouchMove={(event) => {
            if (event.touches.length !== 2 || pinchRef.current === null) {
              return
            }
            const [a, b] = [event.touches[0], event.touches[1]]
            if (!a || !b) {
              return
            }
            const distance = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)
            const delta = distance - pinchRef.current
            pinchRef.current = distance
            setScale((current) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, current + delta / 200)))
          }}
          onTouchEnd={() => {
            pinchRef.current = null
          }}
        >
          <img
            src={src}
            alt={title}
            className="lightbox-image"
            draggable={false}
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            }}
          />
        </div>
      </div>
    </div>
  )
}
