import { describe, expect, it } from 'vitest'
import {
  assertAttachableImage,
  encodeBase64,
  openImageBytes,
  parseImageAttachmentPayload,
  sealImageBytes,
  serializeImageAttachmentPayload,
} from './imageAttachment'
import { formatMessageClock } from './messageClock'
import { previewTextForMessage } from './lastMessagePreview'

describe('imageAttachment payload', () => {
  it('round-trips an image pointer and ignores ordinary text', () => {
    const json = serializeImageAttachmentPayload({
      blobId: '11111111-1111-4111-8111-111111111111',
      fileKey: 'AAAA',
      mime: 'image/jpeg',
      name: 'photo.jpg',
    })
    expect(parseImageAttachmentPayload(json)?.t).toBe('img')
    expect(parseImageAttachmentPayload('hello there')).toBeNull()
  })

  it('rejects video files until that attachment type exists', () => {
    const file = new File([new Uint8Array([1])], 'clip.mp4', { type: 'video/mp4' })
    expect(() => assertAttachableImage(file)).toThrow(/JPEG, PNG, WebP, and GIF/)
  })

  it('seals and opens image bytes with matching associated data', async () => {
    const { initializeSodium } = await import('../crypto/keyExchange')
    await initializeSodium()
    const blobId = '11111111-1111-4111-8111-111111111111'
    const conversationId = '00000000-0000-4000-8000-000000000001'
    const raw = new Uint8Array([1, 2, 3, 4])
    const sealed = await sealImageBytes(raw, conversationId, blobId)
    const opened = await openImageBytes(
      sealed.ciphertext,
      sealed.nonce,
      encodeBase64(sealed.fileKey),
      conversationId,
      blobId,
    )
    expect(Array.from(opened)).toEqual([1, 2, 3, 4])
  })
})

describe('messageClock', () => {
  it('returns an empty string for invalid timestamps', () => {
    expect(formatMessageClock('not-a-date')).toBe('')
    expect(formatMessageClock('')).toBe('')
  })
})

describe('lastMessagePreview', () => {
  it('labels images without exposing pointer JSON', () => {
    const row = previewTextForMessage('{"t":"img"}', 'received', true)
    expect(row.preview).toBe('📷 Photo')
  })
})
