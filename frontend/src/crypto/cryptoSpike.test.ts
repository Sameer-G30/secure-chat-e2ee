// @vitest-environment node

// Import Vitest lifecycle, grouping, and assertion helpers.
import { beforeAll, describe, expect, it } from 'vitest'

// Import only the crypto-spike operations exercised by this proof.
import {
  // Verify authenticated decryption and failure behavior.
  decryptMessage,
  // Derive the sender's client-role directional keys.
  deriveClientSessionKeys,
  // Derive the recipient's complementary server-role keys.
  deriveServerSessionKeys,
  // Produce an authenticated ciphertext envelope.
  encryptMessage,
  // Generate independent endpoint identity keys.
  generateIdentityKeyPair,
  // Initialize libsodium before any synchronous operation.
  initializeSodium,
} from './cryptoSpike'

// Group the complete key-exchange, KDF, and AEAD proof.
describe('libsodium E2EE crypto spike', () => {
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
})
