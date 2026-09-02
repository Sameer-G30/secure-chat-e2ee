// Build a plaintext .txt export of the in-memory, already-decrypted transcript.
// The caller must warn the user first: this writes plaintext to disk, which is
// outside the E2EE trust boundary the rest of the app maintains.

// Import the in-memory bubble shape this formatter reads.
import type { ChatMessage } from '../hooks/useEncryptedConversation'

// Describe one exported line without leaking verification-failed ciphertext.
export function formatTranscriptExport(
  selfUsername: string,
  peerUsername: string,
  messages: ChatMessage[],
): string {
  // Lead with an explicit warning so a later reader of the file sees it first.
  const lines: string[] = [
    `Secure Chat export for ${selfUsername} and ${peerUsername}`,
    'WARNING: This file contains plaintext. It is not end-to-end encrypted on disk.',
    '',
  ]
  for (const message of messages) {
    // Skip rows that never verified; exporting those would invent text we do not have.
    if (message.verificationFailed || message.plaintext === null) {
      continue
    }
    const who = message.direction === 'sent' ? selfUsername : peerUsername
    const edited = message.revision > 0 ? ' (edited)' : ''
    const image = message.attachment
    const body = image ? `[photo: ${image.name}]` : message.plaintext
    lines.push(`${who}${edited}: ${body}`)
  }
  return `${lines.join('\n')}\n`
}

// Trigger a browser download of the formatted transcript.
export function downloadTranscriptFile(filename: string, contents: string): void {
  // Build a blob so the download does not need a server round trip.
  const blob = new Blob([contents], { type: 'text/plain;charset=utf-8' })
  // Create a temporary object URL the anchor can point at.
  const url = URL.createObjectURL(blob)
  // Use a real <a download> click rather than window.open so the filename sticks.
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}
