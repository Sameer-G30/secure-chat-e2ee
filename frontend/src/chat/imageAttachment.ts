// Client-side image-attachment helpers. The file key travels inside a normal
// E2EE chat envelope; the server only stores opaque ciphertext+nonce for the file.

// Import audited libsodium bindings for file AEAD (same primitive as chat envelopes).
import sodium from 'libsodium-wrappers'
// Import base64 helpers already used for envelope ciphertext on the wire.
import { decodeBase64, encodeBase64, initializeSodium } from '../crypto/keyExchange'

// Discriminator stored inside the encrypted chat envelope for an image pointer.
export const IMAGE_ATTACHMENT_TYPE = 'img'
// Cap raw image bytes so the sealed blob stays under the server's 1.5MB ciphertext cap.
export const MAX_IMAGE_BYTES = 1_200_000
// Accept only these media types after decrypt (and when picking a file).
export const ALLOWED_IMAGE_MEDIA_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'] as const

// Describe the JSON payload encrypted as the chat envelope's body for an image.
export interface ImageAttachmentPayload {
  // Mark this envelope as an image pointer rather than ordinary chat text.
  t: typeof IMAGE_ATTACHMENT_TYPE
  // Identify the sealed blob the peer will GET from the conversation.
  blobId: string
  // Carry the 32-byte file key as standard base64 (never sent to the blob table).
  fileKey: string
  // Carry the declared media type so the receiver can build an object URL.
  mime: string
  // Carry the original filename for the bubble label.
  name: string
}

// Describe sealed file bytes ready to POST to /conversations/{id}/blobs.
export interface SealedImageBlob {
  // Carry the client-chosen blob id bound into file associated data before upload.
  blobId: string
  // Carry AEAD ciphertext bytes (including the Poly1305 tag).
  ciphertext: Uint8Array
  // Carry the public nonce required for authenticated decryption.
  nonce: Uint8Array
  // Carry the random 32-byte file key that must travel in the chat envelope.
  fileKey: Uint8Array
}

// Narrow a decrypted chat body into an image pointer, or return null for ordinary text.
export function parseImageAttachmentPayload(plaintext: string): ImageAttachmentPayload | null {
  const trimmed = plaintext.trim()
  if (!trimmed.startsWith('{')) {
    return null
  }
  try {
    const parsed = JSON.parse(trimmed) as {
      t?: unknown
      blobId?: unknown
      fileKey?: unknown
      mime?: unknown
      name?: unknown
    }
    if (parsed.t !== IMAGE_ATTACHMENT_TYPE) {
      return null
    }
    if (
      typeof parsed.blobId !== 'string' ||
      typeof parsed.fileKey !== 'string' ||
      typeof parsed.mime !== 'string' ||
      typeof parsed.name !== 'string'
    ) {
      return null
    }
    if (!ALLOWED_IMAGE_MEDIA_TYPES.includes(parsed.mime as (typeof ALLOWED_IMAGE_MEDIA_TYPES)[number])) {
      return null
    }
    return {
      t: IMAGE_ATTACHMENT_TYPE,
      blobId: parsed.blobId,
      fileKey: parsed.fileKey,
      mime: parsed.mime,
      name: parsed.name,
    }
  } catch {
    return null
  }
}

// Serialize an image pointer so it can be encrypted as an ordinary chat envelope body.
export function serializeImageAttachmentPayload(payload: Omit<ImageAttachmentPayload, 't'>): string {
  const body: ImageAttachmentPayload = {
    t: IMAGE_ATTACHMENT_TYPE,
    blobId: payload.blobId,
    fileKey: payload.fileKey,
    mime: payload.mime,
    name: payload.name,
  }
  return JSON.stringify(body)
}

// Bind file ciphertext to one conversation and blob id (public associated data).
function encodeFileAssociatedData(conversationId: string, blobId: string): string {
  // Canonical JSON is UTF-8 associated data; wrappers accept the string directly.
  return JSON.stringify(['secure-chat-file-v1', conversationId, blobId])
}

// Seal raw image bytes with a random file key using XChaCha20-Poly1305.
export async function sealImageBytes(
  fileBytes: Uint8Array,
  conversationId: string,
  blobId: string,
): Promise<SealedImageBlob> {
  // Wait for the keyExchange sodium singleton used by the rest of the app.
  await initializeSodium()
  // Wait for this module's wrappers too, in case Vitest loaded a second copy.
  await sodium.ready
  // Generate a 32-byte file key that will travel inside the chat envelope, not the blob row.
  const fileKey = sodium.randombytes_buf(sodium.crypto_aead_xchacha20poly1305_ietf_KEYBYTES)
  // Generate a public 24-byte nonce for this one file encryption.
  const nonce = sodium.randombytes_buf(sodium.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES)
  // Bind ciphertext to this conversation and client-chosen blob id.
  const associatedData = encodeFileAssociatedData(conversationId, blobId)
  // Copy into a same-realm Uint8Array so jsdom File buffers satisfy the wrappers.
  const message = new Uint8Array(fileBytes)
  // Seal pixels with the same AEAD used for chat envelopes.
  const ciphertext = sodium.crypto_aead_xchacha20poly1305_ietf_encrypt(
    message,
    associatedData,
    null,
    nonce,
    fileKey,
  )
  return { blobId, ciphertext, nonce, fileKey }
}

// Open sealed image bytes with the file key that arrived inside the chat envelope.
export async function openImageBytes(
  ciphertext: Uint8Array,
  nonce: Uint8Array,
  fileKeyB64: string,
  conversationId: string,
  blobId: string,
): Promise<Uint8Array> {
  // Wait for the keyExchange sodium singleton used by the rest of the app.
  await initializeSodium()
  // Wait for this module's wrappers too, in case Vitest loaded a second copy.
  await sodium.ready
  // Decode the file key that never left the E2EE chat envelope.
  const fileKey = decodeBase64(fileKeyB64)
  // Reconstruct the same public associated data used at seal time.
  const associatedData = encodeFileAssociatedData(conversationId, blobId)
  // Open pixels; a tag mismatch throws and the bubble shows a failed placeholder.
  return sodium.crypto_aead_xchacha20poly1305_ietf_decrypt(
    null,
    new Uint8Array(ciphertext),
    associatedData,
    new Uint8Array(nonce),
    fileKey,
  )
}

// Re-export encodeBase64 so callers can put the file key on the chat envelope JSON.
export { encodeBase64 }

// Guard a picked File before sealing it.
export function assertAttachableImage(file: File): void {
  if (!ALLOWED_IMAGE_MEDIA_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_MEDIA_TYPES)[number])) {
    throw new Error('Only JPEG, PNG, WebP, and GIF images can be attached.')
  }
  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error('That image is too large to attach.')
  }
}
