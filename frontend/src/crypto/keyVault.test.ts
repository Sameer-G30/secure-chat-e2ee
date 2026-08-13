// @vitest-environment node

// Import Vitest lifecycle, grouping, and assertion helpers.
import { beforeAll, describe, expect, it } from 'vitest'

// Import the identity-key generator this vault seals and unseals.
import { generateIdentityKeyPair, initializeSodium } from './keyExchange'
// Import the vault operations under test.
import {
  deleteSealedIdentity,
  hasSealedIdentity,
  sealIdentityKeyPair,
  unsealIdentityKeyPair,
} from './keyVault'

// Group the IndexedDB-backed local key-vault proof.
describe('keyVault', () => {
  // Wait for libsodium's runtime once before all proof cases.
  beforeAll(async () => {
    await initializeSodium()
  })

  // Prove a sealed identity round-trips exactly under the correct password.
  it('unseals the exact original private key under the correct password', async () => {
    const username = 'vault-alice'
    const keyPair = generateIdentityKeyPair()
    await sealIdentityKeyPair(username, 'correct horse battery staple', keyPair)

    const unsealed = await unsealIdentityKeyPair(username, 'correct horse battery staple')
    expect(unsealed.privateKey).toEqual(keyPair.privateKey)
    expect(unsealed.publicKey).toEqual(keyPair.publicKey)
  })

  // Prove a wrong password cannot unseal the private key.
  it('rejects unsealing with an incorrect password', async () => {
    const username = 'vault-bob'
    const keyPair = generateIdentityKeyPair()
    await sealIdentityKeyPair(username, 'the real password', keyPair)

    await expect(unsealIdentityKeyPair(username, 'a wrong password')).rejects.toThrow()
  })

  // Prove the vault correctly reports whether a local identity exists.
  it('reports sealed-identity presence before and after sealing', async () => {
    const username = 'vault-carol'
    expect(await hasSealedIdentity(username)).toBe(false)

    const keyPair = generateIdentityKeyPair()
    await sealIdentityKeyPair(username, 'yet another password', keyPair)
    expect(await hasSealedIdentity(username)).toBe(true)
  })

  // Prove unsealing an account with no local vault entry fails clearly.
  it('rejects unsealing a username with no local vault entry', async () => {
    await expect(unsealIdentityKeyPair('never-sealed-user', 'whatever')).rejects.toThrow(
      /no sealed identity key exists locally/,
    )
  })

  // Prove deleting a sealed identity removes it from local storage.
  it('deletes a sealed identity so it can no longer be unsealed', async () => {
    const username = 'vault-dave'
    const keyPair = generateIdentityKeyPair()
    await sealIdentityKeyPair(username, 'delete-me-password', keyPair)
    expect(await hasSealedIdentity(username)).toBe(true)

    await deleteSealedIdentity(username)
    expect(await hasSealedIdentity(username)).toBe(false)
  })
})
