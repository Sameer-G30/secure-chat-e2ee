// Tests for the symmetric safety number.
import { beforeAll, describe, expect, it } from 'vitest'
import { initializeSodium } from './keyExchange'
import { computeSafetyNumber } from './peerKeyPins'

// Make sure libsodium is ready before hashing.
beforeAll(async () => {
  await initializeSodium()
})

describe('computeSafetyNumber', () => {
  // Two fixed fake public keys.
  const alice = { username: 'alice', publicKey: new Uint8Array(32).fill(1) }
  const bob = { username: 'Bob', publicKey: new Uint8Array(32).fill(2) }

  it('is identical from both sides', () => {
    expect(computeSafetyNumber(alice, bob)).toBe(computeSafetyNumber(bob, alice))
  })

  it('changes when a key changes', () => {
    const swapped = { ...bob, publicKey: new Uint8Array(32).fill(3) }
    expect(computeSafetyNumber(alice, bob)).not.toBe(computeSafetyNumber(alice, swapped))
  })

  it('renders six groups of five digits', () => {
    expect(computeSafetyNumber(alice, bob)).toMatch(/^(\d{5} ){5}\d{5}$/)
  })
})
