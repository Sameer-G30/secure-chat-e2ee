// Import Vitest grouping and assertion helpers.
import { describe, expect, it } from 'vitest'

// Import the formatter under test (the download helper is a thin DOM wrapper).
import { formatTranscriptExport } from './exportTranscript'
import type { ChatMessage } from '../hooks/useEncryptedConversation'

// Build a minimal bubble so tests do not depend on ChatScreen.
function bubble(overrides: Partial<ChatMessage>): ChatMessage {
  return {
    id: '1',
    direction: 'sent',
    plaintext: 'hello',
    verificationFailed: false,
    scamWarning: false,
    clientMessageId: null,
    revision: 0,
    editedAt: null,
    pending: false,
    createdAt: '2026-08-14T12:00:00Z',
    deliveryStatus: 'sent',
    attachment: null,
    ...overrides,
  }
}

describe('formatTranscriptExport', () => {
  it('leads with a plaintext-on-disk warning', () => {
    const text = formatTranscriptExport('alice', 'bob', [])
    expect(text).toContain('WARNING: This file contains plaintext')
  })

  it('skips verification-failed rows instead of inventing text', () => {
    const text = formatTranscriptExport('alice', 'bob', [
      bubble({ plaintext: null, verificationFailed: true }),
      bubble({ direction: 'received', plaintext: 'hi', id: '2' }),
    ])
    expect(text).not.toContain('failed')
    expect(text).toContain('bob: hi')
  })

  it('marks edited messages', () => {
    const text = formatTranscriptExport('alice', 'bob', [
      bubble({ plaintext: 'later', revision: 1 }),
    ])
    expect(text).toContain('alice (edited): later')
  })
})
