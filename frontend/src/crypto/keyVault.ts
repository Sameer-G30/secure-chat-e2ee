// Import only the type of the "sumo" libsodium build; the module itself loads lazily below.
import type SodiumSumo from 'libsodium-wrappers-sumo'

// Import the identity keypair shape produced by keyExchange.ts's key generation.
import type { IdentityKeyPair } from './keyExchange'

// Load the "sumo" libsodium build specifically for crypto_pwhash (Argon2id), on demand.
//
// keyExchange.ts intentionally uses the smaller `libsodium-wrappers` build,
// which covers crypto_kx/crypto_kdf/AEAD but omits crypto_pwhash entirely.
// This module is the only place Argon2id-based key sealing happens, and a
// dynamic import keeps its larger WASM payload out of the main bundle,
// fetched only when a login/registration actually needs to seal or unseal
// a local identity key — mirroring the A6 lazy-load approach used for the
// optional DistilBERT model.
let sodiumPromise: Promise<typeof SodiumSumo> | null = null

async function loadSodium(): Promise<typeof SodiumSumo> {
  if (!sodiumPromise) {
    sodiumPromise = import('libsodium-wrappers-sumo').then(async (module) => {
      const instance = module.default
      await instance.ready
      return instance
    })
  }
  return sodiumPromise
}

// Name the IndexedDB database and object store holding sealed identity keys.
//
// Only the *sealed* (password-encrypted) private key ever touches disk here.
// The plaintext private key exists solely in JS memory for the lifetime of
// one unlocked session and is never sent anywhere, matching the spec's
// "private key must never be transmitted to, or stored on, the server —
// in any form, temporarily or otherwise" requirement.
const DATABASE_NAME = 'secure-chat-key-vault'
const DATABASE_VERSION = 1
const STORE_NAME = 'sealed-identity-keys'

// Describe the at-rest record for one account's password-sealed identity key.
export interface SealedIdentityRecord {
  // Key the IndexedDB record by username so multiple local accounts can coexist.
  username: string
  // Store the public key alongside the seal so callers never need the server
  // just to know their own public identity.
  publicKey: Uint8Array
  // Store the random per-record salt used to derive the wrapping key from the password.
  salt: Uint8Array
  // Store the random nonce used for the AEAD seal.
  nonce: Uint8Array
  // Store the AEAD-sealed private key bytes (ciphertext + authentication tag).
  sealedPrivateKey: Uint8Array
  // Record the crypto_pwhash operation limit used, so unsealing can reproduce it exactly.
  opsLimit: number
  // Record the crypto_pwhash memory limit used, so unsealing can reproduce it exactly.
  memLimit: number
  // Record when this identity was created, for basic local diagnostics only.
  createdAt: number
}

// Open (and lazily create) the single IndexedDB database this module uses.
function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    // Request the versioned database; onupgradeneeded fires only on first use or version bump.
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    // Create the object store the first time this browser opens the vault.
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        // Key records by username; each local browser profile holds one identity per account.
        db.createObjectStore(STORE_NAME, { keyPath: 'username' })
      }
    }
    // Resolve with the ready database handle.
    request.onsuccess = () => resolve(request.result)
    // Reject with IndexedDB's own error so callers see the real failure cause.
    request.onerror = () => reject(request.error ?? new Error('failed to open the key vault'))
  })
}

// Run one read/write transaction against the sealed-identity object store.
async function withStore<T>(
  mode: IDBTransactionMode,
  work: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openDatabase()
  return new Promise<T>((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, mode)
    const store = transaction.objectStore(STORE_NAME)
    const request = work(store)
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('key vault operation failed'))
    // Close the connection once the transaction settles either way.
    transaction.oncomplete = () => db.close()
    transaction.onerror = () => db.close()
  })
}

// Derive a fixed-size symmetric key from the account password and a random salt.
async function deriveWrappingKey(
  password: string,
  salt: Uint8Array,
  opsLimit: number,
  memLimit: number,
): Promise<Uint8Array> {
  const sodium = await loadSodium()
  // Use libsodium's Argon2id-backed crypto_pwhash, matching §6.1's "password-derived,
  // client-side Argon2id" requirement for sealing the local key vault.
  return sodium.crypto_pwhash(
    sodium.crypto_aead_xchacha20poly1305_ietf_KEYBYTES,
    password,
    salt,
    opsLimit,
    memLimit,
    sodium.crypto_pwhash_ALG_DEFAULT,
  )
}

// Encrypt a freshly generated identity keypair's private half for local storage.
export async function sealIdentityKeyPair(
  username: string,
  password: string,
  keyPair: IdentityKeyPair,
): Promise<void> {
  const sodium = await loadSodium()
  // Generate a fresh salt for every seal so the same password never derives the same key twice.
  const salt = sodium.randombytes_buf(sodium.crypto_pwhash_SALTBYTES)
  // Use libsodium's INTERACTIVE limits: strong enough for at-rest local storage while staying
  // fast in a browser tab; documented explicitly as a trade-off in the project README.
  const opsLimit = sodium.crypto_pwhash_OPSLIMIT_INTERACTIVE
  const memLimit = sodium.crypto_pwhash_MEMLIMIT_INTERACTIVE
  const wrappingKey = await deriveWrappingKey(password, salt, opsLimit, memLimit)
  // Generate a fresh public nonce for this seal, reusing the approved AEAD primitive (A1).
  const nonce = sodium.randombytes_buf(sodium.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES)
  // Bind the seal to the owning username so a copied record cannot be unsealed under another.
  const associatedData = sodium.from_string(username)
  const sealedPrivateKey = sodium.crypto_aead_xchacha20poly1305_ietf_encrypt(
    keyPair.privateKey,
    associatedData,
    null,
    nonce,
    wrappingKey,
  )
  // Best-effort zero the derived wrapping key now that sealing is complete.
  wrappingKey.fill(0)

  const record: SealedIdentityRecord = {
    username,
    publicKey: keyPair.publicKey,
    salt,
    nonce,
    sealedPrivateKey,
    opsLimit,
    memLimit,
    createdAt: Date.now(),
  }
  // IndexedDB's structured clone algorithm stores Uint8Array values directly; no base64 needed.
  await withStore('readwrite', (store) => store.put(record))
}

// Decrypt a previously sealed private key using the account password.
export async function unsealIdentityKeyPair(
  username: string,
  password: string,
): Promise<IdentityKeyPair> {
  const sodium = await loadSodium()
  const record = await withStore<SealedIdentityRecord | undefined>('readonly', (store) =>
    store.get(username),
  )
  if (!record) {
    // No local vault entry exists for this account on this browser/device.
    throw new RangeError(`no sealed identity key exists locally for "${username}"`)
  }
  const wrappingKey = await deriveWrappingKey(
    password,
    record.salt,
    record.opsLimit,
    record.memLimit,
  )
  const associatedData = sodium.from_string(username)
  try {
    // A wrong password produces a wrapping key that fails Poly1305 verification here.
    const privateKey = sodium.crypto_aead_xchacha20poly1305_ietf_decrypt(
      null,
      record.sealedPrivateKey,
      associatedData,
      record.nonce,
      wrappingKey,
    )
    return { publicKey: record.publicKey, privateKey }
  } finally {
    // Best-effort zero the derived wrapping key regardless of success or failure.
    wrappingKey.fill(0)
  }
}

// Check locally whether this browser already holds a sealed identity for a username.
export async function hasSealedIdentity(username: string): Promise<boolean> {
  const record = await withStore<SealedIdentityRecord | undefined>('readonly', (store) =>
    store.get(username),
  )
  return record !== undefined
}

// Remove a locally sealed identity, e.g. on explicit "forget this device" actions.
export async function deleteSealedIdentity(username: string): Promise<void> {
  await withStore('readwrite', (store) => store.delete(username))
}
