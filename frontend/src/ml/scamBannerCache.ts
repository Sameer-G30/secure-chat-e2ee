// Persist last on-device scam-banner decisions so a reload can paint warnings
// immediately. Ciphertext history has no scores; WASM DistilBERT takes seconds to
// recreate. This cache stores only {warned, revision, checkpointId} keyed by the
// server message UUID — never plaintext, never tokens.

// Import the catalog id type so a DistilBERT cache row cannot be applied to TF-IDF.
import type { CheckpointId } from './types'

// Prefix localStorage keys so banners stay per signed-in username.
export const BANNER_CACHE_PREFIX = 'secure-chat-scam-banners:'

// Schema version so a future layout can ignore stale blobs instead of throwing.
// v2 drops v1 rows: conversation-window scoring and the ChatScreen 0.35 overlay
// would otherwise reuse banners computed from a single isolated DM.
const CACHE_VERSION = 2

// Cap stored rows so a long-lived tab cannot grow localStorage without bound.
const MAX_CACHED_BANNERS = 500

// One cached decision for one message id at one edit revision.
export interface CachedBanner {
  // True when that checkpoint showed the scam banner.
  warned: boolean
  // Ignore the row after an edit (plaintext changed; revision advanced).
  revision: number
  // Ignore the row when the operator switched DistilBERT / LSTM / TF-IDF.
  checkpointId: CheckpointId
}

// On-disk JSON blob for one username.
interface BannerCacheFile {
  // Must match CACHE_VERSION.
  v: number
  // Map of server (or temporary local) message id → last decision.
  entries: Record<string, CachedBanner>
}

// Build the per-user localStorage key.
export function bannerCacheKey(username: string): string {
  // Tokens never go in this key; only the handle is stored.
  return `${BANNER_CACHE_PREFIX}${username}`
}

// Read and validate the blob; return an empty map on any failure.
export function loadBannerCache(username: string): Record<string, CachedBanner> {
  try {
    // Read the per-user key.
    const raw = window.localStorage.getItem(bannerCacheKey(username))
    // Missing key means this device has never scored for this handle.
    if (!raw) {
      // Callers treat an empty map as all cache misses.
      return {}
    }
    // Parse JSON; invalid JSON falls into the catch below.
    const parsed = JSON.parse(raw) as BannerCacheFile
    // Reject an unknown schema rather than guessing field names.
    if (parsed.v !== CACHE_VERSION || typeof parsed.entries !== 'object' || parsed.entries === null) {
      // Treat corruption as a cold start.
      return {}
    }
    // Hand back the map (values are checked again in readCachedBanner).
    return parsed.entries
  } catch {
    // Private mode or a truncated write must not break chat.
    return {}
  }
}

// Write the whole map back (small; simpler than patching one key in place).
function saveBannerCache(username: string, entries: Record<string, CachedBanner>): void {
  try {
    // Drop oldest-inserted keys when over the cap (JSON object insertion order).
    const ids = Object.keys(entries)
    // Only trim when the operator has scored hundreds of DMs on this device.
    if (ids.length > MAX_CACHED_BANNERS) {
      // Remove the surplus from the front of the key list.
      for (const extraId of ids.slice(0, ids.length - MAX_CACHED_BANNERS)) {
        // Delete does not store plaintext; it only forgets a UUID.
        delete entries[extraId]
      }
    }
    // Persist warned/revision/checkpointId only.
    window.localStorage.setItem(
      bannerCacheKey(username),
      JSON.stringify({ v: CACHE_VERSION, entries } satisfies BannerCacheFile),
    )
  } catch {
    // Quota or private mode: in-memory banners still work for this session.
  }
}

// Return the cached warned flag, or null when the row must be classified.
export function readCachedBanner(
  username: string,
  messageId: string,
  revision: number,
  checkpointId: CheckpointId,
): boolean | null {
  // Look up this message id in the per-user map.
  const entry = loadBannerCache(username)[messageId]
  // Missing id, other checkpoint, or an older/newer revision is a miss.
  if (!entry || entry.checkpointId !== checkpointId || entry.revision !== revision) {
    // Caller should run the ready model (or wait for DistilBERT to load).
    return null
  }
  // Same graph + same revision: reuse last session's banner without WASM.
  return entry.warned
}

// Store one classify() result so the next reload can paint immediately.
export function writeCachedBanner(
  username: string,
  messageId: string,
  revision: number,
  checkpointId: CheckpointId,
  warned: boolean,
): void {
  // Load-modify-save; the map is small.
  const entries = loadBannerCache(username)
  // Overwrite any previous decision for this id.
  entries[messageId] = { warned, revision, checkpointId }
  // Persist without plaintext.
  saveBannerCache(username, entries)
}

// Follow the local-id → server-id rename from an "accepted" ack.
export function renameCachedBanner(username: string, fromId: string, toId: string): void {
  // No-op when the server reused the same UUID.
  if (fromId === toId) {
    // Nothing to rewrite.
    return
  }
  // Load the map so we can move one key.
  const entries = loadBannerCache(username)
  // Read the temporary-id row written while the send was pending.
  const entry = entries[fromId]
  // Classify may still be in flight; skip if nothing was stored yet.
  if (!entry) {
    // The in-flight result will write under the old id; handleAccepted races are rare.
    return
  }
  // Store under the server id used by ciphertext history after reload.
  entries[toId] = entry
  // Drop the temporary key so it cannot leak as a second row.
  delete entries[fromId]
  // Persist the rename.
  saveBannerCache(username, entries)
}
