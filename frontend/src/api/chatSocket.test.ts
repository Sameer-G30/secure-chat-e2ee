// Import Vitest grouping and assertion helpers.
import { describe, expect, it, vi } from 'vitest'

// Import the relay-frame parsers used by ChatScreen's WebSocket client.
import { handleRelayJson, parseEpochFrame, parseRelayedEnvelope } from './chatSocket'
import type { ChatSocketHandlers } from './chatSocket'

// Group Slice 8 epoch-metadata parsing separately from the envelope parser.
describe('chatSocket epoch frames', () => {
  // Prove a well-formed bump returns the public counter and nothing else.
  it('parseEpochFrame reads current_epoch from a metadata-only frame', () => {
    // Build the exact JSON shape the backend broadcasts after a bump.
    const frame = { type: 'epoch', current_epoch: 1 }
    // Require the integer clients will pass to deriveEpochKey on the next encrypt.
    expect(parseEpochFrame(frame)).toBe(1)
  })

  // Prove an epoch frame is never mistaken for a ciphertext envelope.
  it('parseRelayedEnvelope ignores epoch metadata', () => {
    // The bump frame has no ciphertext fields.
    const frame = { type: 'epoch', current_epoch: 1 }
    // Envelope parsing must return null so handleRelayJson takes the epoch branch.
    expect(parseRelayedEnvelope(frame)).toBeNull()
  })

  // Prove handleRelayJson updates onEpoch and does not report an unreadable envelope.
  it('handleRelayJson routes epoch frames to onEpoch without a protocol error', () => {
    // Capture each callback so we can assert the bump is metadata-only.
    const onEnvelope = vi.fn()
    // A false unreadable-envelope error would flash in the chat UI on every bump.
    const onProtocolError = vi.fn()
    // Close is unused for this dispatch.
    const onClose = vi.fn()
    // Epoch is the callback under test.
    const onEpoch = vi.fn()
    // Assemble the handler bag ChatScreen registers.
    const handlers: ChatSocketHandlers = {
      onEnvelope,
      onProtocolError,
      onClose,
      onEpoch,
    }
    // Dispatch the same JSON the WebSocket onmessage path would parse.
    handleRelayJson({ type: 'epoch', current_epoch: 3 }, handlers)
    // The in-memory encrypt epoch must move to 3.
    expect(onEpoch).toHaveBeenCalledWith(3)
    // Ciphertext handling must not run.
    expect(onEnvelope).not.toHaveBeenCalled()
    // A bump must not look like a malformed envelope.
    expect(onProtocolError).not.toHaveBeenCalled()
  })

  // Prove a draft-shaped field on an epoch frame is ignored (server never sends one).
  it('handleRelayJson does not treat extra epoch fields as plaintext', () => {
    // If a buggy client included draft text, the handler still only reads current_epoch.
    const onEpoch = vi.fn()
    // Protocol errors would mean the extra field broke routing.
    const onProtocolError = vi.fn()
    // Dispatch a frame that must still be classified as epoch metadata.
    handleRelayJson(
      { type: 'epoch', current_epoch: 1, draft: 'must never be shown' },
      {
        onEnvelope: vi.fn(),
        onProtocolError,
        onClose: vi.fn(),
        onEpoch,
      },
    )
    // The counter is still applied.
    expect(onEpoch).toHaveBeenCalledWith(1)
    // No protocol error from the extra field.
    expect(onProtocolError).not.toHaveBeenCalled()
  })
})
