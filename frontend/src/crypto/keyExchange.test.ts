// @vitest-environment node

// Import Vitest lifecycle, grouping, and assertion helpers.
import { beforeAll, describe, expect, it } from 'vitest'

// Import only the key-exchange operations exercised by this proof.
import {
  // Verify authenticated decryption and failure behavior.
  decryptMessage,
  // Round-trip raw bytes through the base64 transport encoding.
  decodeBase64,
  // Decide which crypto_kx role an endpoint plays for a pairing.
  determineSessionRole,
  // Derive the sender's client-role directional keys.
  deriveClientSessionKeys,
  // Derive the recipient's complementary server-role keys.
  deriveServerSessionKeys,
  // Derive directional keys without the caller branching on role.
  deriveSessionKeys,
  // Round-trip raw bytes through the base64 transport encoding.
  encodeBase64,
  // Produce an authenticated ciphertext envelope.
  encryptMessage,
  // Generate independent endpoint identity keys.
  generateIdentityKeyPair,
  // Initialize libsodium before any synchronous operation.
  initializeSodium,
} from './keyExchange'

// Group the complete key-exchange, KDF, and AEAD proof.
describe('keyExchange', () => {
  // Wait for libsodium's runtime once before all proof cases.
  beforeAll(async () => {
    // Ensure every cryptographic binding is initialized.
    await initializeSodium()
  })

  // Prove complementary directional keys recover the original plaintext.
  it('round-trips an authenticated message through client and server roles', () => {
    // Generate Alice's client-role identity entirely in this endpoint simulation.
    const alice = generateIdentityKeyPair()
    // Generate Bob's server-role identity independently.
    const bob = generateIdentityKeyPair()
    // Derive Alice's directional keys from Bob's public key.
    const aliceSession = deriveClientSessionKeys(alice, bob.publicKey)
    // Derive Bob's complementary directional keys from Alice's public key.
    const bobSession = deriveServerSessionKeys(bob, alice.publicKey)
    // Define public metadata that the authentication tag must bind.
    const metadata = {
      // Use a stable conversation identifier for this proof.
      conversationId: '00000000-0000-4000-8000-000000000001',
      // Use Alice's stable sender identifier for this proof.
      senderId: '00000000-0000-4000-8000-000000000002',
    }
    // Encrypt a message with Alice's transmit direction and epoch seven.
    const envelope = encryptMessage('verify me end to end', aliceSession.transmitKey, 7, metadata)
    // Decrypt with Bob's mathematically complementary receive direction.
    const recovered = decryptMessage(envelope, bobSession.receiveKey, metadata)
    // Require exact Unicode plaintext recovery.
    expect(recovered).toBe('verify me end to end')
  })

  // Prove Poly1305 authentication rejects modified ciphertext.
  it('rejects a tampered ciphertext instead of returning corrupted text', () => {
    // Generate independent endpoint identities for the tamper proof.
    const alice = generateIdentityKeyPair()
    // Generate Bob's independent recipient identity.
    const bob = generateIdentityKeyPair()
    // Derive Alice's sender direction.
    const aliceSession = deriveClientSessionKeys(alice, bob.publicKey)
    // Derive Bob's matching recipient direction.
    const bobSession = deriveServerSessionKeys(bob, alice.publicKey)
    // Define metadata authenticated by the original envelope.
    const metadata = {
      // Bind the envelope to one test conversation.
      conversationId: '00000000-0000-4000-8000-000000000003',
      // Bind the envelope to Alice's test account.
      senderId: '00000000-0000-4000-8000-000000000004',
    }
    // Encrypt a valid envelope before simulating an untrusted relay.
    const envelope = encryptMessage('tampering must fail', aliceSession.transmitKey, 2, metadata)
    // Copy ciphertext so the original envelope remains unchanged.
    const tamperedCiphertext = new Uint8Array(envelope.ciphertext)
    // Flip one payload bit to invalidate the authentication tag.
    tamperedCiphertext[0] ^= 1
    // Create a tampered envelope with the original nonce and epoch.
    const tamperedEnvelope = {
      // Supply the attacker-modified bytes.
      ciphertext: tamperedCiphertext,
      // Reuse the original public nonce.
      nonce: envelope.nonce,
      // Reuse the original epoch identifier.
      keyEpoch: envelope.keyEpoch,
    }
    // Require decryption to throw instead of exposing unauthenticated output.
    expect(() => decryptMessage(tamperedEnvelope, bobSession.receiveKey, metadata)).toThrow()
  })

  // Prove associated data prevents cross-conversation replay.
  it('rejects ciphertext replayed with different conversation metadata', () => {
    // Generate Alice's identity for the replay proof.
    const alice = generateIdentityKeyPair()
    // Generate Bob's identity for the replay proof.
    const bob = generateIdentityKeyPair()
    // Derive Alice's sender direction from Bob's public key.
    const aliceSession = deriveClientSessionKeys(alice, bob.publicKey)
    // Derive Bob's recipient direction from Alice's public key.
    const bobSession = deriveServerSessionKeys(bob, alice.publicKey)
    // Define the legitimate metadata authenticated by Alice.
    const originalMetadata = {
      // Identify the intended conversation.
      conversationId: '00000000-0000-4000-8000-000000000005',
      // Identify the intended sender.
      senderId: '00000000-0000-4000-8000-000000000006',
    }
    // Encrypt a valid envelope for the intended conversation.
    const envelope = encryptMessage('do not replay me', aliceSession.transmitKey, 0, originalMetadata)
    // Define altered metadata for an attacker-selected conversation.
    const replayedMetadata = {
      // Change only the conversation identifier.
      conversationId: '00000000-0000-4000-8000-000000000007',
      // Preserve the claimed sender to isolate conversation binding.
      senderId: originalMetadata.senderId,
    }
    // Require authentication failure under the substituted conversation.
    expect(() => decryptMessage(envelope, bobSession.receiveKey, replayedMetadata)).toThrow()
  })

  // Prove old envelopes still open after a later epoch subkey is derived (Slice 8).
  it('decrypts an epoch-0 envelope after deriving an epoch-1 subkey', () => {
    // Generate Alice's identity for the two-epoch round trip.
    const alice = generateIdentityKeyPair()
    // Generate Bob's complementary identity.
    const bob = generateIdentityKeyPair()
    // Derive Alice's sender direction from Bob's public key.
    const aliceSession = deriveClientSessionKeys(alice, bob.publicKey)
    // Derive Bob's recipient direction from Alice's public key.
    const bobSession = deriveServerSessionKeys(bob, alice.publicKey)
    // Bind both envelopes to the same conversation and sender.
    const metadata = {
      // Use a stable conversation identifier for this proof.
      conversationId: '00000000-0000-4000-8000-000000000008',
      // Use Alice's stable sender identifier for this proof.
      senderId: '00000000-0000-4000-8000-000000000009',
    }
    // Encrypt under epoch 0 the way history rows keep their original key_epoch.
    const oldEnvelope = encryptMessage('history from epoch zero', aliceSession.transmitKey, 0, metadata)
    // Encrypt a later send under epoch 1 without deleting the epoch-0 subkey.
    const newEnvelope = encryptMessage('fresh from epoch one', aliceSession.transmitKey, 1, metadata)
    // History decrypt uses the envelope's keyEpoch, not "whatever current is".
    expect(decryptMessage(oldEnvelope, bobSession.receiveKey, metadata)).toBe(
      'history from epoch zero',
    )
    // The new send uses the bumped subkey id.
    expect(decryptMessage(newEnvelope, bobSession.receiveKey, metadata)).toBe('fresh from epoch one')
    // Mixing the epoch id with the other envelope must fail authentication.
    expect(() =>
      decryptMessage(
        { ...oldEnvelope, keyEpoch: 1 },
        bobSession.receiveKey,
        metadata,
      ),
    ).toThrow()
  })

  // Prove role derivation is a stable, symmetric, and deterministic rule.
  describe('determineSessionRole', () => {
    // Prove the lexicographically earlier username always plays "client".
    it('assigns client to the lexicographically earlier username', () => {
      // "alice" sorts before "bob" under ordinary string comparison.
      expect(determineSessionRole('alice', 'bob')).toBe('client')
      // The same pairing viewed from Bob's side must assign the complementary role.
      expect(determineSessionRole('bob', 'alice')).toBe('server')
    })

    // Prove identical usernames cannot produce a meaningful role.
    it('rejects deriving a role against your own username', () => {
      // A conversation with oneself is not a valid crypto_kx pairing.
      expect(() => determineSessionRole('alice', 'alice')).toThrow(RangeError)
    })

    // Prove empty identifiers are rejected before any comparison happens.
    it('rejects empty usernames', () => {
      // An empty peer username cannot be compared meaningfully.
      expect(() => determineSessionRole('alice', '')).toThrow(RangeError)
      // An empty self username cannot be compared meaningfully either.
      expect(() => determineSessionRole('', 'bob')).toThrow(RangeError)
    })
  })

  // Prove the role-agnostic wrapper reaches the same keys as the manual calls.
  it('deriveSessionKeys matches the manually role-dispatched directional keys', () => {
    // Generate two independent identities named so their sort order is fixed.
    const alice = generateIdentityKeyPair()
    const bob = generateIdentityKeyPair()
    // Derive Alice's keys through the convenience wrapper.
    const aliceKeys = deriveSessionKeys(
      { keys: alice, username: 'alice' },
      { publicKey: bob.publicKey, username: 'bob' },
    )
    // Derive Alice's keys through the manual role-specific call for comparison.
    const expectedAliceKeys = deriveClientSessionKeys(alice, bob.publicKey)
    // Require both derivation paths to agree exactly.
    expect(aliceKeys.transmitKey).toEqual(expectedAliceKeys.transmitKey)
    expect(aliceKeys.receiveKey).toEqual(expectedAliceKeys.receiveKey)

    // Derive Bob's keys through the convenience wrapper from the opposite side.
    const bobKeys = deriveSessionKeys(
      { keys: bob, username: 'bob' },
      { publicKey: alice.publicKey, username: 'alice' },
    )
    // Derive Bob's keys through the manual role-specific call for comparison.
    const expectedBobKeys = deriveServerSessionKeys(bob, alice.publicKey)
    // Require both derivation paths to agree exactly.
    expect(bobKeys.transmitKey).toEqual(expectedBobKeys.transmitKey)
    expect(bobKeys.receiveKey).toEqual(expectedBobKeys.receiveKey)
  })

  // Prove base64 transport encoding round-trips exactly for wire-format use.
  it('round-trips arbitrary bytes through base64 transport encoding', () => {
    // Generate real key-sized random bytes rather than a trivial fixed string.
    const original = generateIdentityKeyPair().publicKey
    // Encode the bytes as the format that will travel over JSON/WebSocket.
    const encoded = encodeBase64(original)
    // Confirm the transport form is a plain string, not binary.
    expect(typeof encoded).toBe('string')
    // Decode and require an exact match with the original bytes.
    expect(decodeBase64(encoded)).toEqual(original)
  })
})
