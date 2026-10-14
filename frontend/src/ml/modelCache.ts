// Cache API for the DistilBERT default graph and its WordPiece sidecar.
// Keys include the checkpoint id, file name, and the manifest byte length so a
// new export misses. Plaintext messages are never stored here. This is not an
// HTTP cache header on the unhashed model.onnx name.

// Cache storage name for this browser origin's DistilBERT default artifacts.
const DISTILBERT_CACHE_NAME = 'secure-chat-distilbert-default'

// Build a Request the Cache API can store. The host is synthetic so it cannot
// collide with a year-long HTTP cache of /ml/.../model.onnx.
export function distilbertCacheRequest(filename: string, versionBytes: number): Request {
  // Version is the manifest's onnx byte length (the sidecar has no separate hash).
  const url = `https://secure-chat.invalid/ml/distilbert_default/${filename}?bytes=${versionBytes}`
  // GET is the only method the Cache API match uses for this artifact.
  return new Request(url)
}

// Read one artifact from the Cache API, or fetch it and store the bytes.
export async function readOrFetchBytes(
  networkUrl: string,
  filename: string,
  versionBytes: number,
): Promise<ArrayBuffer> {
  // The versioned key changes when manifest.onnx_bytes changes.
  const cacheKey = distilbertCacheRequest(filename, versionBytes)
  // jsdom and locked-down browsers may not expose the Cache API.
  if (typeof caches !== 'undefined') {
    try {
      // Open this origin's DistilBERT cache (created on first use).
      const cache = await caches.open(DISTILBERT_CACHE_NAME)
      // Look up this file at this byte length.
      const hit = await cache.match(cacheKey)
      // A hit means this browser already paid for the download.
      if (hit) {
        // Return the cached bytes without touching the network.
        return await hit.arrayBuffer()
      }
    } catch {
      // A cache read failure falls through to the network fetch.
    }
  }
  // First visit, or a cache miss after the export's byte length changed.
  const response = await fetch(networkUrl)
  // A missing export is a load failure, not an empty graph.
  if (!response.ok) {
    // Include the URL so an operator can see which artifact is missing.
    throw new Error(`Failed to fetch ${networkUrl} (${response.status})`)
  }
  // Copy the body once so both ORT and the cache can read it.
  const buffer = await response.arrayBuffer()
  // Store a copy. arrayBuffer() already consumed the network body.
  if (typeof caches !== 'undefined') {
    try {
      // Re-open in case the earlier open failed after a transient error.
      const cache = await caches.open(DISTILBERT_CACHE_NAME)
      // slice(0) so a later ORT decode cannot detach the cached copy.
      await cache.put(
        cacheKey,
        new Response(buffer.slice(0), {
          headers: { 'Content-Type': 'application/octet-stream' },
        }),
      )
    } catch {
      // Quota or a private window: this session still uses the fetched bytes.
    }
  }
  // Hand the original buffer to the caller.
  return buffer
}
