// Persist one restorable chat session in IndexedDB so a reload can reopen
// ChatScreen without asking for the password.
//
// This record holds the username, the current single-use refresh token, and
// the unsealed X25519 keypair. The account password is never written. The
// access token is not written: startup calls POST /auth/refresh once and
// keeps the new access token in memory. Nothing here goes into localStorage
// or sessionStorage. Logout deletes the record.

// Name the database so it stays separate from the Argon2id key vault.
const DATABASE_NAME = 'secure-chat-session'
// Bump this if the record shape changes. v1 is one session row.
const DATABASE_VERSION = 1
// Name the object store that holds the single current session.
const STORE_NAME = 'restorable-session'
// Key the only row. This app keeps one signed-in account per page load.
const RECORD_ID = 'current'

// Describe the IndexedDB row a reload can turn back into a session.
export interface RestorableSessionRecord {
  // Key the single row. Callers always use RECORD_ID.
  id: typeof RECORD_ID
  // Identify the account ChatScreen should open.
  username: string
  // Carry the refresh token the next POST /auth/refresh must present.
  refreshToken: string
  // Carry the unsealed X25519 public key. The sealed vault still has a copy.
  publicKey: Uint8Array
  // Carry the unsealed X25519 private key so reload does not need the password.
  privateKey: Uint8Array
}

// Open (and lazily create) the session database.
function openDatabase(): Promise<IDBDatabase> {
  // Return a promise so callers can await the IDB open request.
  return new Promise((resolve, reject) => {
    // Request the versioned database; onupgradeneeded fires on first use.
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    // Create the object store the first time this browser opens the database.
    request.onupgradeneeded = () => {
      // Read the database handle from the upgrade event.
      const db = request.result
      // Create the store only when this version has not already created it.
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        // Key the row by the fixed id field.
        db.createObjectStore(STORE_NAME, { keyPath: 'id' })
      }
    }
    // Surface an open failure rather than hanging startup.
    request.onerror = () => {
      reject(request.error ?? new Error('Could not open session storage.'))
    }
    // Hand the ready database to the caller.
    request.onsuccess = () => {
      resolve(request.result)
    }
  })
}

// Read the restorable session, or null when this browser is signed out.
export async function readRestorableSession(): Promise<RestorableSessionRecord | null> {
  // Open the database for this one read.
  const db = await openDatabase()
  // Wait for the get request to settle.
  const row = await new Promise<RestorableSessionRecord | undefined>((resolve, reject) => {
    // Read the single row.
    const tx = db.transaction(STORE_NAME, 'readonly')
    // Look up the fixed key.
    const request = tx.objectStore(STORE_NAME).get(RECORD_ID)
    // Resolve with the row, or undefined when logout already deleted it.
    request.onsuccess = () => {
      resolve(request.result as RestorableSessionRecord | undefined)
    }
    // Surface a read failure so startup can fall back to AuthScreen.
    request.onerror = () => {
      reject(request.error ?? new Error('Could not read the saved session.'))
    }
  })
  // Close the connection so the next test can delete the database.
  db.close()
  // No row means the next screen is AuthScreen.
  if (!row) {
    return null
  }
  // Copy key bytes out of the IndexedDB view so later mutations cannot alias it.
  return {
    id: RECORD_ID,
    username: row.username,
    refreshToken: row.refreshToken,
    publicKey: new Uint8Array(row.publicKey),
    privateKey: new Uint8Array(row.privateKey),
  }
}

// Write the restorable session. Replaces any previous row.
export async function writeRestorableSession(input: {
  username: string
  refreshToken: string
  publicKey: Uint8Array
  privateKey: Uint8Array
}): Promise<void> {
  // Build the row. Password and access token are intentionally absent.
  const record: RestorableSessionRecord = {
    id: RECORD_ID,
    username: input.username,
    refreshToken: input.refreshToken,
    publicKey: new Uint8Array(input.publicKey),
    privateKey: new Uint8Array(input.privateKey),
  }
  // Open the database for this one write.
  const db = await openDatabase()
  // Wait until the put transaction completes.
  await new Promise<void>((resolve, reject) => {
    // Replace the single row.
    const tx = db.transaction(STORE_NAME, 'readwrite')
    // Resolve only after the transaction commits.
    tx.oncomplete = () => {
      resolve()
    }
    // Surface a write failure so the caller can drop the session.
    tx.onerror = () => {
      reject(tx.error ?? new Error('Could not save the session.'))
    }
    // Store the row under the fixed key.
    tx.objectStore(STORE_NAME).put(record)
  })
  // Close the connection.
  db.close()
}

// Delete the restorable session. Logout is the only product path that calls this.
export async function deleteRestorableSession(): Promise<void> {
  // Open the database for this one delete.
  const db = await openDatabase()
  // Wait until the delete transaction completes.
  await new Promise<void>((resolve, reject) => {
    // Remove the single row.
    const tx = db.transaction(STORE_NAME, 'readwrite')
    // Resolve after the transaction commits, even if the row was already gone.
    tx.oncomplete = () => {
      resolve()
    }
    // Surface a delete failure so logout can still clear React state.
    tx.onerror = () => {
      reject(tx.error ?? new Error('Could not delete the saved session.'))
    }
    // Delete by the fixed key.
    tx.objectStore(STORE_NAME).delete(RECORD_ID)
  })
  // Close the connection.
  db.close()
}
