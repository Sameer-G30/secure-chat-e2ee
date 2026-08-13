// Import the key-upload API call that publishes only the public half of an identity.
import { uploadMyPublicKey } from '../api/keysClient'
// Import base64 transport helpers and keypair generation from the message-crypto module.
import { decodeBase64, encodeBase64, generateIdentityKeyPair, initializeSodium } from './keyExchange'
import type { IdentityKeyPair } from './keyExchange'
// Import the local password-sealed vault this module reads and writes.
import { hasSealedIdentity, sealIdentityKeyPair, unsealIdentityKeyPair } from './keyVault'

// Describe the outcome of ensuring a usable local identity exists after login.
export interface IdentitySetupResult {
  // Carry the base64 public key now known to match both the server and the local vault.
  publicKeyBase64: string
  // Carry the unsealed keypair for immediate in-memory use this session (never persisted as-is).
  keyPair: IdentityKeyPair
  // Tell the caller whether a brand-new identity was generated during this call.
  generatedNewKeyPair: boolean
}

// Raise this specifically for the unsupported "known account, unknown device" case.
export class IdentitySetupError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'IdentitySetupError'
  }
}

// Reconcile the server's key-upload state with this browser's local key vault.
//
// Implements §6.1 of the spec ("on registration or first login on a new
// device, generate an X25519 keypair... Persist the private key in
// IndexedDB, encrypted at rest with a key derived from the user's
// password") together with A6-adjacent scoping: multi-device recovery is
// explicitly out of scope, so a known account with no local vault entry
// fails loudly instead of silently generating a second, unlinked identity.
export async function ensureIdentityKeys(
  username: string,
  password: string,
  accessToken: string,
  hasPublicKeyOnServer: boolean,
): Promise<IdentitySetupResult> {
  await initializeSodium()
  const sealedLocally = await hasSealedIdentity(username)

  if (hasPublicKeyOnServer && sealedLocally) {
    // Common case: an existing user logging in again on the same browser.
    const keyPair = await unsealIdentityKeyPair(username, password)
    return { publicKeyBase64: encodeBase64(keyPair.publicKey), keyPair, generatedNewKeyPair: false }
  }

  if (hasPublicKeyOnServer && !sealedLocally) {
    // A real account with a server-side key, but this browser has no matching vault
    // entry — e.g. a new device, cleared site data, or a different browser profile.
    // Multi-device key synchronization/recovery is explicitly out of scope (see the
    // README threat model); fail clearly rather than minting an unlinked identity
    // that could never decrypt messages sent to the account's real public key.
    throw new IdentitySetupError(
      `No local key vault was found for "${username}" on this device. ` +
        'Multi-device key recovery is not supported yet, so this browser cannot send or ' +
        'receive end-to-end encrypted messages for this account until that ships.',
    )
  }

  if (!hasPublicKeyOnServer && sealedLocally) {
    // A locally sealed identity exists, but the server has no record of its public
    // key — most likely a prior upload that failed after registration. Recover by
    // re-uploading the same local public key rather than generating a new identity.
    const keyPair = await unsealIdentityKeyPair(username, password)
    const publicKeyBase64 = encodeBase64(keyPair.publicKey)
    await uploadMyPublicKey(accessToken, publicKeyBase64)
    return { publicKeyBase64, keyPair, generatedNewKeyPair: false }
  }

  // Neither the server nor this browser has a key yet: this is a genuinely first-time
  // setup. Generate a fresh X25519 identity, seal it locally, then publish the public half.
  const keyPair = generateIdentityKeyPair()
  await sealIdentityKeyPair(username, password, keyPair)
  const publicKeyBase64 = encodeBase64(keyPair.publicKey)
  await uploadMyPublicKey(accessToken, publicKeyBase64)
  return { publicKeyBase64, keyPair, generatedNewKeyPair: true }
}

// Re-export decodeBase64 for callers that need to compare/parse a fetched peer key.
export { decodeBase64 }
