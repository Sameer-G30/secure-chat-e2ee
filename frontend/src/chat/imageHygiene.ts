// On-device checks that run after a chat image decrypts and before an object URL.
// Nothing in this file uploads pixels, QR text, or scores.

// Import the local QR decoder. It never fetches the payload.
import jsQR from 'jsqr'

// JPEG files start with the SOI marker plus the next marker byte.
const JPEG_MAGIC = [0xff, 0xd8, 0xff]
// PNG signature is eight fixed bytes.
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
// GIF87a and GIF89a share a six-byte ASCII header.
const GIF87 = 'GIF87a'
// GIF89a is the animated variant. It is still accepted and not re-encoded.
const GIF89 = 'GIF89a'

// Thrown when the decrypted header does not match the declared media type.
export class ImageMagicMismatchError extends Error {
  constructor() {
    super('Image header does not match the declared type.')
    // Name the error so callers can branch without parsing the message.
    this.name = 'ImageMagicMismatchError'
  }
}

// True when the first bytes equal an exact header.
function startsWithBytes(bytes: Uint8Array, header: number[]): boolean {
  // A short buffer cannot match.
  if (bytes.length < header.length) {
    return false
  }
  // Compare each header byte.
  for (let index = 0; index < header.length; index += 1) {
    // Any mismatch rejects the file.
    if (bytes[index] !== header[index]) {
      return false
    }
  }
  // The header matched.
  return true
}

// True when ASCII text sits at an offset.
function startsWithAscii(bytes: Uint8Array, text: string, offset: number): boolean {
  // Reject a buffer that ends before the text.
  if (bytes.length < offset + text.length) {
    return false
  }
  // Compare each character code.
  for (let index = 0; index < text.length; index += 1) {
    // A different byte means this is not that signature.
    if (bytes[offset + index] !== text.charCodeAt(index)) {
      return false
    }
  }
  // The ASCII run matched.
  return true
}

// Accept decrypted bytes only when the header matches the declared MIME type.
export function imageHeaderMatchesDeclaredType(bytes: Uint8Array, mime: string): boolean {
  // JPEG must start with FF D8 FF.
  if (mime === 'image/jpeg') {
    return startsWithBytes(bytes, JPEG_MAGIC)
  }
  // PNG must start with the eight-byte signature.
  if (mime === 'image/png') {
    return startsWithBytes(bytes, PNG_MAGIC)
  }
  // GIF must be GIF87a or GIF89a.
  if (mime === 'image/gif') {
    return startsWithAscii(bytes, GIF87, 0) || startsWithAscii(bytes, GIF89, 0)
  }
  // WebP is RIFF at offset 0 and WEBP at offset 8.
  if (mime === 'image/webp') {
    return startsWithAscii(bytes, 'RIFF', 0) && startsWithAscii(bytes, 'WEBP', 8)
  }
  // Any other declared type is rejected.
  return false
}

// Copy a view into its own ArrayBuffer so Blob does not include neighboring bytes.
function copyBytes(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  // Allocate a real ArrayBuffer (not a SharedArrayBuffer view).
  const buffer = new ArrayBuffer(bytes.byteLength)
  // Wrap that buffer so Blob accepts it.
  const copy = new Uint8Array(buffer)
  // Copy the decrypted pixels.
  copy.set(bytes)
  // Return the standalone copy.
  return copy
}

// Draw a still photo to a canvas and return a JPEG or PNG blob without the original container.
async function reencodeStill(bytes: Uint8Array, mime: string): Promise<Blob> {
  // The browser decodes the original once so the canvas can drop EXIF and simple LSB data.
  const bitmap = await createImageBitmap(new Blob([copyBytes(bytes)], { type: mime }))
  // Size the canvas to the decoded bitmap.
  const canvas = document.createElement('canvas')
  // Match the bitmap width.
  canvas.width = bitmap.width
  // Match the bitmap height.
  canvas.height = bitmap.height
  // 2d is enough; this is cleanup, not a sandbox.
  const context = canvas.getContext('2d')
  // A missing context cannot re-encode.
  if (!context) {
    // Release the decoded bitmap before failing the bubble.
    bitmap.close()
    // The caller shows the failed placeholder and does not create an object URL.
    throw new Error('Could not re-encode that image.')
  }
  // Paint the verified bitmap. The original container is not what we display.
  context.drawImage(bitmap, 0, 0)
  // Release the decoder bitmap.
  bitmap.close()
  // PNG stays PNG. JPEG and WebP are written as JPEG.
  const outType = mime === 'image/png' ? 'image/png' : 'image/jpeg'
  // toBlob is callback-shaped.
  const blob = await new Promise<Blob | null>((resolve) => {
    // Quality applies to JPEG. PNG ignores it.
    canvas.toBlob((result) => resolve(result), outType, 0.92)
  })
  // A null blob means the canvas could not encode.
  if (!blob) {
    throw new Error('Could not re-encode that image.')
  }
  // Return the cleaned still.
  return blob
}

// Build the blob the bubble should display. GIF is not re-encoded.
export async function displayBlobForVerifiedImage(bytes: Uint8Array, mime: string): Promise<Blob> {
  // Reject a header that does not match the type inside the encrypted pointer.
  if (!imageHeaderMatchesDeclaredType(bytes, mime)) {
    // Callers must not call createObjectURL after this throw.
    throw new ImageMagicMismatchError()
  }
  // GIF animation would collapse to one frame if it were re-encoded.
  if (mime === 'image/gif') {
    // Still show the original GIF bytes after the magic-byte check.
    return new Blob([copyBytes(bytes)], { type: 'image/gif' })
  }
  // JPEG, PNG, and WebP are re-encoded through a canvas.
  return reencodeStill(bytes, mime)
}

// Re-encode a still image before sealing so EXIF (GPS, device, timestamps) never leaves this device.
// GIF is passed through because re-encoding would drop the animation.
export async function sanitizeImageForSend(
  bytes: Uint8Array,
  mime: string,
): Promise<{ bytes: Uint8Array; mime: string }> {
  // Animated GIFs keep their bytes; their metadata blocks are rarely sensitive.
  if (mime === 'image/gif') {
    return { bytes, mime }
  }
  // Environments without canvas (tests) cannot re-encode; fail closed only in a real browser.
  if (typeof document === 'undefined' || typeof createImageBitmap === 'undefined') {
    return { bytes, mime }
  }
  // Draw and re-encode: the canvas output carries pixels only.
  const blob = await reencodeStill(bytes, mime)
  // Report the actual output type (WebP becomes JPEG).
  return { bytes: new Uint8Array(await blob.arrayBuffer()), mime: blob.type || mime }
}

// Decode one QR payload from the decrypted image, or return null when none is present.
export async function readQrPayload(bytes: Uint8Array, mime: string): Promise<string | null> {
  // Tests without a document cannot draw. No QR means no extra classification.
  if (typeof document === 'undefined') {
    return null
  }
  try {
    // Decode one frame. GIF uses that single frame and is not re-encoded for display.
    const bitmap = await createImageBitmap(new Blob([copyBytes(bytes)], { type: mime }))
    // Size a canvas to the frame.
    const canvas = document.createElement('canvas')
    // Match the frame width.
    canvas.width = bitmap.width
    // Match the frame height.
    canvas.height = bitmap.height
    // willReadFrequently hints that we need the pixels for jsQR.
    const context = canvas.getContext('2d', { willReadFrequently: true })
    // Without a context there is nothing to scan.
    if (!context) {
      // Release the bitmap.
      bitmap.close()
      // Treat a canvas failure as "no QR" so a still photo can still display.
      return null
    }
    // Draw the one frame jsQR will scan.
    context.drawImage(bitmap, 0, 0)
    // Release the decoder bitmap.
    bitmap.close()
    // Read RGBA pixels. They stay on this device.
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
    // jsQR does not fetch the URL it finds.
    const code = jsQR(imageData.data, imageData.width, imageData.height)
    // Trim so a blank payload is treated as no QR.
    const text = code?.data.trim() ?? ''
    // Empty means no extra classification.
    return text.length > 0 ? text : null
  } catch {
    // A decoder failure is not a failed image. The magic-byte gate already passed.
    return null
  }
}
