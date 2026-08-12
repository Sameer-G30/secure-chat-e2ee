// Import audited libsodium bindings rather than implementing primitives locally.
import sodium from 'libsodium-wrappers'

// Use the exact eight-byte domain-separation context required by libsodium KDF.
const MESSAGE_KEY_CONTEXT = 'msgkey01'

// Describe the client-generated long-term X25519 key material this module produces.
export interface IdentityKeyPair {
  // Share this public key through the authenticated key-upload API (Slice 3).
  publicKey: Uint8Array
  // Keep this private key exclusively on the user's endpoint, sealed at rest (Slice 3).
  privateKey: Uint8Array
}

// Describe directional keys returned by crypto_kx.
export interface DirectionalSessionKeys {
  // Decrypt messages received from the peer with this key.
  receiveKey: Uint8Array
  // Encrypt messages sent to the peer with this key.
  transmitKey: Uint8Array
}

// Describe all data the ciphertext-only server may store and relay.
export interface EncryptedEnvelope {
  // Carry authenticated encrypted bytes without plaintext.
  ciphertext: Uint8Array
  // Carry the unique public nonce required for decryption.
  nonce: Uint8Array
  // Identify which locally derived epoch key protects this envelope.
  keyEpoch: number
}

// Describe metadata cryptographically bound to each encrypted envelope.
export interface EnvelopeMetadata {
  // Prevent an envelope from authenticating in another conversation.
  conversationId: string
  // Prevent an envelope from authenticating under another sender identity.
  senderId: string
}

// Describe the two crypto_kx roles a session pairing can take.
//
// §6.3 of the spec: role is decided by comparing usernames lexicographically.
// The comparison only needs to be a stable, symmetric rule both peers agree
// on independently — it carries no meaning beyond breaking the tie.
export type SessionRole = 'client' | 'server'

// Wait until the WebAssembly or JavaScript sodium runtime is initialized.
export async function initializeSodium(): Promise<void> {
  // Resolve only after every requested cryptographic function is available.
  await sodium.ready
}

// Generate an X25519 keypair entirely on the current endpoint.
export function generateIdentityKeyPair(): IdentityKeyPair {
  // Ask libsodium to generate correctly sized random key material.
  const keyPair = sodium.crypto_kx_keypair()
  // Expose explicit names that discourage accidental private-key upload.
  return {
    // Return the shareable X25519 public key.
    publicKey: keyPair.publicKey,
    // Return the endpoint-only X25519 private key.
    privateKey: keyPair.privateKey,
  }
}

// Decide which crypto_kx role this endpoint plays for one pairing of accounts.
export function determineSessionRole(selfUsername: string, peerUsername: string): SessionRole {
  // Reject inputs that cannot produce a meaningful, stable comparison.
  if (!selfUsername || !peerUsername) {
    // Fail loudly rather than silently deriving an arbitrary role.
    throw new RangeError('determineSessionRole requires two non-empty usernames')
  }
  // A conversation with oneself has no well-defined client/server pairing.
  if (selfUsername === peerUsername) {
    // Refuse rather than returning an arbitrary, meaningless role.
    throw new RangeError('determineSessionRole requires two distinct usernames')
  }
  // Whoever sorts first plays the crypto_kx "client" role for this pairing.
  return selfUsername < peerUsername ? 'client' : 'server'
}

// Derive client-role directional session keys using both public identities.
export function deriveClientSessionKeys(
  // Accept the client-role user's complete local keypair.
  clientKeys: IdentityKeyPair,
  // Accept only the server-role peer's public key.
  serverPublicKey: Uint8Array,
): DirectionalSessionKeys {
  // Derive independent receive and transmit keys through crypto_kx.
  const sessionKeys = sodium.crypto_kx_client_session_keys(
    // Bind derivation to the client's public identity.
    clientKeys.publicKey,
    // Prove the client's identity with its endpoint-only private key.
    clientKeys.privateKey,
    // Bind derivation to the selected peer's public identity.
    serverPublicKey,
  )
  // Convert libsodium names into application-level directional names.
  return {
    // The client receives data encrypted with the server's transmit key.
    receiveKey: sessionKeys.sharedRx,
    // The client transmits data decryptable with the server's receive key.
    transmitKey: sessionKeys.sharedTx,
  }
}

// Derive server-role directional session keys using both public identities.
export function deriveServerSessionKeys(
  // Accept the server-role user's complete local keypair.
  serverKeys: IdentityKeyPair,
  // Accept only the client-role peer's public key.
  clientPublicKey: Uint8Array,
): DirectionalSessionKeys {
  // Derive complementary receive and transmit keys through crypto_kx.
  const sessionKeys = sodium.crypto_kx_server_session_keys(
    // Bind derivation to the server-role user's public identity.
    serverKeys.publicKey,
    // Prove the server-role user's identity with its local private key.
    serverKeys.privateKey,
    // Bind derivation to the client-role peer's public identity.
    clientPublicKey,
  )
  // Convert libsodium names into application-level directional names.
  return {
    // The server role receives data encrypted with the client's transmit key.
    receiveKey: sessionKeys.sharedRx,
    // The server role transmits data decryptable with the client's receive key.
    transmitKey: sessionKeys.sharedTx,
  }
}

// Derive directional session keys without the caller needing to branch on role.
//
// This is the production entry point real conversation code should call: it
// takes the raw ingredients (both usernames and both public identities) and
// internally applies determineSessionRole plus the matching crypto_kx call.
export function deriveSessionKeys(
  // Accept the local user's complete keypair and username.
  self: { keys: IdentityKeyPair; username: string },
  // Accept the remote peer's public key and username.
  peer: { publicKey: Uint8Array; username: string },
): DirectionalSessionKeys {
  // Decide the role once, from the same rule every peer applies independently.
  const role = determineSessionRole(self.username, peer.username)
  // Dispatch to the crypto_kx call matching the derived role.
  return role === 'client'
    ? deriveClientSessionKeys(self.keys, peer.publicKey)
    : deriveServerSessionKeys(self.keys, peer.publicKey)
}

// Derive a fixed-size message key for one non-secret epoch number.
export function deriveEpochKey(
  // Accept the directional crypto_kx session key as KDF master material.
  sessionKey: Uint8Array,
  // Accept the server-coordinated non-negative epoch identifier.
  keyEpoch: number,
): Uint8Array {
  // Reject values that cannot safely identify a libsodium subkey.
  if (!Number.isSafeInteger(keyEpoch) || keyEpoch < 0) {
    // Fail before invoking cryptography with ambiguous epoch input.
    throw new RangeError('keyEpoch must be a non-negative safe integer')
  }
  // Derive an independent XChaCha20 key under the approved context.
  return sodium.crypto_kdf_derive_from_key(
    // Match the key length required by the approved AEAD primitive.
    sodium.crypto_aead_xchacha20poly1305_ietf_KEYBYTES,
    // Use the epoch as libsodium's deterministic subkey identifier.
    keyEpoch,
    // Separate chat message keys from every other possible KDF use.
    MESSAGE_KEY_CONTEXT,
    // Derive from one directional crypto_kx session key.
    sessionKey,
  )
}

// Serialize authenticated metadata with an unambiguous versioned representation.
function encodeAssociatedData(
  // Accept metadata the server can route but must not be able to alter.
  metadata: EnvelopeMetadata,
  // Accept the exact epoch used to derive the message key.
  keyEpoch: number,
): Uint8Array {
  // Encode a JSON array so field boundaries cannot be confused by concatenation.
  const canonicalMetadata = JSON.stringify([
    // Version the associated-data format for future migrations.
    'secure-chat-envelope-v1',
    // Bind the encrypted bytes to one conversation.
    metadata.conversationId,
    // Bind the encrypted bytes to the claimed sender.
    metadata.senderId,
    // Bind the encrypted bytes to one epoch derivation.
    keyEpoch,
  ])
  // Convert the canonical Unicode string with libsodium's own UTF-8 helper.
  return sodium.from_string(canonicalMetadata)
}

// Encrypt one plaintext message into a relay-safe authenticated envelope.
export function encryptMessage(
  // Accept plaintext only inside the sender endpoint.
  plaintext: string,
  // Accept the sender's directional transmit key.
  transmitSessionKey: Uint8Array,
  // Accept the current non-secret epoch counter.
  keyEpoch: number,
  // Accept routing metadata that must be authenticated.
  metadata: EnvelopeMetadata,
): EncryptedEnvelope {
  // Derive a dedicated message key for this epoch and direction.
  const epochKey = deriveEpochKey(transmitSessionKey, keyEpoch)
  // Generate a fresh full-length public nonce for every encryption.
  const nonce = sodium.randombytes_buf(
    // Use the exact nonce size required by XChaCha20-Poly1305 IETF.
    sodium.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES,
  )
  // Convert message text into endpoint-only UTF-8 bytes through libsodium.
  const plaintextBytes = sodium.from_string(plaintext)
  // Authenticate conversation, sender, and epoch without encrypting routing data.
  const associatedData = encodeAssociatedData(metadata, keyEpoch)
  // Encrypt and authenticate the message through the approved AEAD primitive.
  const ciphertext = sodium.crypto_aead_xchacha20poly1305_ietf_encrypt(
    // Supply endpoint-only plaintext bytes.
    plaintextBytes,
    // Bind the canonical routing metadata to the authentication tag.
    associatedData,
    // Use no secret nonce because the protocol does not require one.
    null,
    // Supply the unique public nonce carried in the envelope.
    nonce,
    // Supply the per-epoch directional encryption key.
    epochKey,
  )
  // Return exactly the fields the future server may relay and store.
  return {
    // Carry authenticated ciphertext and its Poly1305 tag.
    ciphertext,
    // Carry the public nonce needed for recipient verification.
    nonce,
    // Carry the non-secret derivation identifier.
    keyEpoch,
  }
}

// Verify and decrypt one authenticated envelope on the recipient endpoint.
export function decryptMessage(
  // Accept the ciphertext-only envelope received from the server.
  envelope: EncryptedEnvelope,
  // Accept the recipient's complementary directional receive key.
  receiveSessionKey: Uint8Array,
  // Accept routing metadata expected by the recipient UI.
  metadata: EnvelopeMetadata,
): string {
  // Derive the same directional epoch key used by the sender.
  const epochKey = deriveEpochKey(receiveSessionKey, envelope.keyEpoch)
  // Recreate the exact associated-data bytes expected by the authentication tag.
  const associatedData = encodeAssociatedData(metadata, envelope.keyEpoch)
  // Verify the tag and recover bytes, throwing on any mismatch or tampering.
  const plaintextBytes = sodium.crypto_aead_xchacha20poly1305_ietf_decrypt(
    // Use no secret nonce because encryption supplied none.
    null,
    // Verify the received ciphertext and attached authentication tag.
    envelope.ciphertext,
    // Require exact conversation, sender, and epoch metadata.
    associatedData,
    // Supply the public nonce delivered with the envelope.
    envelope.nonce,
    // Supply the recipient's complementary per-epoch key.
    epochKey,
  )
  // Decode verified UTF-8 bytes into UI-safe React text.
  return sodium.to_string(plaintextBytes)
}

// Encode arbitrary bytes for JSON/WebSocket transport and for the key-upload API.
//
// The wire format for ciphertext, nonces, and public keys is base64 text, not
// raw bytes, because JSON and REST payloads cannot carry Uint8Array directly.
export function encodeBase64(bytes: Uint8Array): string {
  // Use libsodium's constant-width base64 encoding rather than a hand-rolled one.
  return sodium.to_base64(bytes, sodium.base64_variants.ORIGINAL)
}

// Decode base64 transport text back into raw bytes for cryptographic use.
export function decodeBase64(encoded: string): Uint8Array {
  // Mirror encodeBase64's variant so round-tripping is always exact.
  return sodium.from_base64(encoded, sodium.base64_variants.ORIGINAL)
}
