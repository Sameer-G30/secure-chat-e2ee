// Unit tests for the local scam-banner cache (no plaintext, per-username keys).

// Import Vitest helpers used by the rest of the frontend unit suite.
import { afterEach, describe, expect, it } from 'vitest'

// Import the cache helpers under test.
import {
  BANNER_CACHE_PREFIX,
  readCachedBanner,
  renameCachedBanner,
  writeCachedBanner,
} from './scamBannerCache'

// Drop alice's blob so tests cannot leak into each other.
afterEach(() => {
  // localStorage is shared across the jsdom document.
  window.localStorage.clear()
})

// Describe hit/miss and the pending-id rename.
describe('scamBannerCache', () => {
  // DistilBERT default must not reuse a TF-IDF or stale-revision row.
  it('returns a DistilBERT default hit and ignores a TF-IDF row', () => {
    // Pretend last session warned on this server id.
    writeCachedBanner('alice', 'msg-1', 0, 'distilbert_default', true)
    // Same checkpoint + revision is a hit.
    expect(readCachedBanner('alice', 'msg-1', 0, 'distilbert_default')).toBe(true)
    // Eager TF-IDF must not inherit DistilBERT banners.
    expect(readCachedBanner('alice', 'msg-1', 0, 'tfidf_best')).toBeNull()
    // An edit advances revision and must re-score.
    expect(readCachedBanner('alice', 'msg-1', 1, 'distilbert_default')).toBeNull()
    // Ciphertext plaintext must never be written next to the UUID.
    expect(window.localStorage.getItem(`${BANNER_CACHE_PREFIX}alice`)).not.toContain('hello')
  })

  // History after reload uses the server id, not the optimistic local id.
  it('renames a pending local id to the server id after accept', () => {
    // Classify finished before the accepted ack.
    writeCachedBanner('alice', 'local-1', 0, 'distilbert_default', false)
    // Mirror handleAccepted's local → server rewrite.
    renameCachedBanner('alice', 'local-1', 'server-1')
    // The temporary id must not survive as a second cache row.
    expect(readCachedBanner('alice', 'local-1', 0, 'distilbert_default')).toBeNull()
    // Reload of ciphertext history looks up the server id.
    expect(readCachedBanner('alice', 'server-1', 0, 'distilbert_default')).toBe(false)
  })
})
