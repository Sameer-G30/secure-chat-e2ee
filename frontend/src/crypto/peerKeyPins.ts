// Trust-on-first-use pinning of peer identity keys plus a human-comparable safety number.
// Purpose: a malicious or compromised directory can swap a peer's public key; pinning detects it.

// Import audited libsodium bindings for hashing (no hand-rolled primitives).
import sodium from 'libsodium-wrappers'

// Name the IndexedDB database that holds pins, separate from the vault and session stores.
const DATABASE_NAME = 'secure-chat-peer-pins'
// Bump if the record shape changes.
const DATABASE_VERSION = 1
// Name the object store keyed by "<self>|<peer>".
const STORE_NAME = 'pins'

// Describe the result of comparing a fetched peer key with the stored pin.
export type PinStatus = 'new' | 'match' | 'changed'

// Build the storage key so two accounts on one browser never share pins.
function pinKey(selfUsername: string, peerUsername: string): string {
  // Lowercase both names to match the server's canonical username handling.
  return `${selfUsername.toLowerCase()}|${peerUsername.toLowerCase()}`
}

// Format 30 hash bytes as twelve groups of five digits (Signal-style readable code).
function formatDigits(bytes: Uint8Array): string {
  // Collect the groups as strings.
  const groups: string[] = []
  // Use six groups of five digits from 3 bytes each, giving a 30-digit code.
  for (let i = 0; i < 30; i += 5) {
    // Combine 5 bytes into a number below 2^40 safely within double precision.
    let value = 0
    for (let j = 0; j < 5; j += 1) {
      // Shift by multiplication to avoid 32-bit overflow.
      value = value * 256 + (bytes[i + j] ?? 0)
    }
    // Reduce to five decimal digits and left-pad.
    groups.push(String(value % 100000).padStart(5, '0'))
  }
  // Join into a spaced code that is easy to read aloud.
  return groups.join(' ')
}

// Compute the safety number both peers will see identically.
// Inputs are sorted by username so the order of "self" and "peer" does not matter.
export function computeSafetyNumber(
  a: { username: string; publicKey: Uint8Array },
  b: { username: string; publicKey: Uint8Array },
): string {
  // Sort the two identities by lowercase username for a symmetric result.
  const [first, second] = [a, b].sort((x, y) =>
    x.username.toLowerCase() < y.username.toLowerCase() ? -1 : 1,
  )
  // Encode "username NUL key" for each side so fields cannot be confused.
  const encoder = new TextEncoder()
  const left = new Uint8Array([...encoder.encode(first.username.toLowerCase()), 0, ...first.publicKey])
  const right = new Uint8Array([...encoder.encode(second.username.toLowerCase()), 0, ...second.publicKey])
  // Concatenate both sides into one message.
  const message = new Uint8Array(left.length + right.length)
  message.set(left, 0)
  message.set(right, left.length)
  // BLAKE2b with a 32-byte output, via libsodium generic hash.
  const digest = sodium.crypto_generichash(32, message, null)
  // Render as digit groups.
  return formatDigits(digest)
}

// Open (and lazily create) the pin database.
function openDatabase(): Promise<IDBDatabase> {
  // Wrap the callback API in a promise.
  return new Promise((resolve, reject) => {
    // Request the versioned database.
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    // Create the store on first use.
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        // Rows are { id, publicKey } keyed by id.
        request.result.createObjectStore(STORE_NAME, { keyPath: 'id' })
      }
    }
    // Surface open failures.
    request.onerror = () => reject(request.error ?? new Error('Could not open pin storage.'))
    // Hand back the ready database.
    request.onsuccess = () => resolve(request.result)
  })
}

// Read the pinned base64 key, or null when this peer was never seen.
async function readPin(id: string): Promise<string | null> {
  // Open the database for this read.
  const db = await openDatabase()
  try {
    // Wait for the single get to settle.
    return await new Promise<string | null>((resolve, reject) => {
      const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(id)
      request.onsuccess = () => resolve((request.result as { publicKey: string } | undefined)?.publicKey ?? null)
      request.onerror = () => reject(request.error ?? new Error('Could not read pin.'))
    })
  } finally {
    // Always close so tests can delete the database.
    db.close()
  }
}

// Write (or replace) the pinned key.
async function writePin(id: string, publicKey: string): Promise<void> {
  // Open the database for this write.
  const db = await openDatabase()
  try {
    // Wait for the transaction to complete.
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite')
      tx.objectStore(STORE_NAME).put({ id, publicKey })
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error ?? new Error('Could not write pin.'))
    })
  } finally {
    // Always close the connection.
    db.close()
  }
}

// Compare a fetched peer key with the pin; pin it automatically on first sight.
export async function checkPeerKey(
  selfUsername: string,
  peerUsername: string,
  peerPublicKeyBase64: string,
): Promise<PinStatus> {
  // Look up the stored pin.
  const id = pinKey(selfUsername, peerUsername)
  const pinned = await readPin(id)
  // First contact: trust on first use and remember.
  if (pinned === null) {
    await writePin(id, peerPublicKeyBase64)
    return 'new'
  }
  // Same key as before.
  if (pinned === peerPublicKeyBase64) {
    return 'match'
  }
  // The directory now serves a different key: caller must block until the user accepts.
  return 'changed'
}

// Replace the pin after the user explicitly accepts the new key.
export async function acceptPeerKey(
  selfUsername: string,
  peerUsername: string,
  peerPublicKeyBase64: string,
): Promise<void> {
  // Overwrite the old pin with the accepted key.
  await writePin(pinKey(selfUsername, peerUsername), peerPublicKeyBase64)
}
