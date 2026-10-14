// Magic-byte gate for decrypted chat images. Canvas re-encode is not required here.

// Import Vitest helpers.
import { describe, expect, it } from 'vitest'

// Import the header check under test.
import { imageHeaderMatchesDeclaredType } from './imageHygiene'

// Build a buffer that starts with the given bytes.
function bytes(header: number[]): Uint8Array {
  // Pad so short signatures are not the whole file.
  return new Uint8Array([...header, 0x00, 0x01])
}

describe('imageHeaderMatchesDeclaredType', () => {
  it('accepts JPEG, PNG, GIF87a, GIF89a, and WebP headers', () => {
    // JPEG starts with FF D8 FF.
    expect(imageHeaderMatchesDeclaredType(bytes([0xff, 0xd8, 0xff]), 'image/jpeg')).toBe(true)
    // PNG's eight-byte signature.
    expect(
      imageHeaderMatchesDeclaredType(
        bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        'image/png',
      ),
    ).toBe(true)
    // GIF87a is a still GIF.
    expect(
      imageHeaderMatchesDeclaredType(
        new Uint8Array([0x47, 0x49, 0x46, 0x38, 0x37, 0x61, 0x00]),
        'image/gif',
      ),
    ).toBe(true)
    // GIF89a is the animated header and is still accepted.
    expect(
      imageHeaderMatchesDeclaredType(
        new Uint8Array([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x00]),
        'image/gif',
      ),
    ).toBe(true)
    // WebP is RIFF at 0 and WEBP at 8.
    const webp = new Uint8Array(16)
    webp.set([0x52, 0x49, 0x46, 0x46], 0)
    webp.set([0x57, 0x45, 0x42, 0x50], 8)
    expect(imageHeaderMatchesDeclaredType(webp, 'image/webp')).toBe(true)
  })

  it('rejects a header that does not match the declared type', () => {
    // A PNG signature declared as JPEG must not become an object URL.
    const png = bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    expect(imageHeaderMatchesDeclaredType(png, 'image/jpeg')).toBe(false)
    // An unknown declared type is rejected even with a JPEG header.
    expect(imageHeaderMatchesDeclaredType(bytes([0xff, 0xd8, 0xff]), 'image/bmp')).toBe(false)
  })
})
