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
