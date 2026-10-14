// Format bubble clocks from ISO timestamps the server already sends on envelopes.

// Render a short wall-clock time for a message bubble, or an empty string if invalid.
export function formatMessageClock(iso: string | null | undefined): string {
  if (!iso) {
    return ''
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return ''
  }
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// Local calendar day key so two bubbles on the same day share one separator.
export function localCalendarDayKey(iso: string | null | undefined): string | null {
  // Missing timestamps do not invent a date label.
  if (!iso) {
    return null
  }
  // Parse the envelope's createdAt. Invalid strings produce no separator.
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return null
  }
  // Local year/month/day. This label is not sent to the server.
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`
}

// Short local date label shown when the calendar day changes between bubbles.
export function formatLocalDateSeparator(iso: string): string {
  // The caller already checked that the timestamp parses.
  const date = new Date(iso)
  // Weekday plus calendar date in the browser's locale.
  return date.toLocaleDateString([], {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}
