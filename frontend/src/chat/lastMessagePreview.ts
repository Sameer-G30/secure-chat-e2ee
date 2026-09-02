// Persist last-message previews on this device only. The server never returns
// preview text because it cannot decrypt envelopes.

// Name the IndexedDB database that stores per-conversation preview rows.
const DB_NAME = 'secure-chat-previews'
// Bump this if the object-store shape changes; v1 is a single store of preview rows.
const DB_VERSION = 1
// Name the object store keyed by `${username}:${conversationId}`.
const STORE_NAME = 'previews'

// Describe one locally stored last-message preview.
export interface LastMessagePreview {
  // Carry a short decrypted snippet, or "📷 Photo" for an image attachment.
  preview: string
  // Carry when this preview was written, as milliseconds since epoch.
  at: number
  // Distinguish sent vs received so the sidebar can prefix "You:" when useful.
  direction: 'sent' | 'received'
}

// Open (or create) the preview database.
function openPreviewDb(): Promise<IDBDatabase> {
  // Return a promise so callers can await the IDB open request.
  return new Promise((resolve, reject) => {
    // Open the named database at the current schema version.
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    // Create the object store the first time this browser sees the database.
    request.onupgradeneeded = () => {
      // Read the database handle from the upgrade event.
      const db = request.result
      // Create the store only when this version has not already created it.
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        // Key previews by the composite owner+conversation string.
        db.createObjectStore(STORE_NAME)
      }
    }
    // Surface an open failure rather than hanging the sidebar render.
    request.onerror = () => {
      reject(request.error ?? new Error('Could not open preview storage.'))
    }
    // Hand the ready database to the caller.
    request.onsuccess = () => {
      resolve(request.result)
    }
  })
}

// Build the IndexedDB key for one owner's preview of one peer.
export function previewStorageKey(username: string, peerUsername: string): string {
  // Namespace by the signed-in handle and the peer handle the sidebar already has.
  return `${username}:peer:${peerUsername}`
}

// Write a last-message preview after a successful local decrypt or send.
export async function writeLastMessagePreview(
  username: string,
  peerUsername: string,
  preview: LastMessagePreview,
): Promise<void> {
  // Skip storage when this tab is in the logout render.
  if (!username || !peerUsername) {
    return
  }
  const db = await openPreviewDb()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.oncomplete = () => {
      resolve()
    }
    tx.onerror = () => {
      reject(tx.error ?? new Error('Could not write preview.'))
    }
    tx.objectStore(STORE_NAME).put(preview, previewStorageKey(username, peerUsername))
  })
  db.close()
}

// Read one peer's last-message preview, or null when none is stored.
export async function readLastMessagePreview(
  username: string,
  peerUsername: string,
): Promise<LastMessagePreview | null> {
  if (!username || !peerUsername) {
    return null
  }
  const db = await openPreviewDb()
  const row = await new Promise<LastMessagePreview | null>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const request = tx.objectStore(STORE_NAME).get(previewStorageKey(username, peerUsername))
    request.onsuccess = () => {
      resolve((request.result as LastMessagePreview | undefined) ?? null)
    }
    request.onerror = () => {
      reject(request.error ?? new Error('Could not read preview.'))
    }
  })
  db.close()
  return row
}

// Load every preview for one signed-in handle into a map keyed by peer username.
export async function readAllLastMessagePreviews(
  username: string,
): Promise<Map<string, LastMessagePreview>> {
  const result = new Map<string, LastMessagePreview>()
  if (!username) {
    return result
  }
  const db = await openPreviewDb()
  const prefix = `${username}:peer:`
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.openCursor()
    request.onsuccess = () => {
      const cursor = request.result
      if (!cursor) {
        resolve()
        return
      }
      const key = String(cursor.key)
      if (key.startsWith(prefix)) {
        result.set(key.slice(prefix.length), cursor.value as LastMessagePreview)
      }
      cursor.continue()
    }
    request.onerror = () => {
      reject(request.error ?? new Error('Could not list previews.'))
    }
  })
  db.close()
  return result
}

// Build the sidebar snippet for a verified message, hiding image JSON from the list.
export function previewTextForMessage(
  plaintext: string,
  direction: 'sent' | 'received',
  isImage: boolean,
): LastMessagePreview {
  const snippet = isImage
    ? '📷 Photo'
    : plaintext.length > 80
      ? `${plaintext.slice(0, 80)}…`
      : plaintext
  return {
    preview: direction === 'sent' ? `You: ${snippet}` : snippet,
    at: Date.now(),
    direction,
  }
}
