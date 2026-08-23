// ONNX Runtime Web session helpers: one heavy (transformer/LSTM) graph at a time.

// Import the WASM-only build so we do not pull WebGL/WebGPU backends.
import * as ort from 'onnxruntime-web/wasm'
// Resolve the SIMD-threaded binary through the package export (not public/, not /dist/).
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
// Resolve the Emscripten factory through the package export so Vite can transform it.
import ortWasmMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'

// Import checkpoint path helpers.
import { checkpointAssetUrl } from './paths'
import type { CheckpointFamily, CheckpointId } from './types'

// Point ORT at Vite-processed URLs. A `/ort/` prefix makes Vite import public/ JS and fail.
ort.env.wasm.wasmPaths = {
  // Override the .wasm fetch path with the hashed Vite asset URL.
  wasm: ortWasmUrl,
  // Override the .mjs factory path so the worker never imports `/ort/*.mjs`.
  mjs: ortWasmMjsUrl,
}
// Run InferenceSession.run in an ORT-owned Web Worker so DistilBERT does not freeze the composer.
ort.env.wasm.proxy = true
// Enable SIMD when the browser supports it (Chrome/Edge on this machine).
ort.env.wasm.simd = true
// Use extra WASM threads only when COOP/COEP made the page cross-origin isolated.
ort.env.wasm.numThreads =
  typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
    ? Math.min(4, Math.max(1, navigator.hardwareConcurrency || 1))
    : 1

// The one allowed DistilBERT or LSTM session; TF-IDF may stay loaded beside it.
let heavySession: { id: CheckpointId; family: CheckpointFamily; session: ort.InferenceSession } | null =
  null

// TF-IDF logistic-head session (small); A5 allows this to stay eager.
let tfidfSession: { id: CheckpointId; session: ort.InferenceSession } | null = null

// Create an InferenceSession from a fetched ArrayBuffer.
export async function createSessionFromBuffer(buffer: ArrayBuffer): Promise<ort.InferenceSession> {
  // WASM execution provider is the portable CPU path for all six graphs.
  return ort.InferenceSession.create(buffer, {
    // WASM stays the portable CPU path; WebGPU is skipped because int8 DistilBERT is unreliable there.
    executionProviders: ['wasm'],
  })
}

// Fetch an ONNX file as an ArrayBuffer from Vite's public/ml folder.
export async function fetchOnnxBuffer(id: CheckpointId, filename: string): Promise<ArrayBuffer> {
  // Build the public URL for this checkpoint's graph.
  const url = checkpointAssetUrl(id, filename)
  // Fetch the raw bytes; Vite serves gitignored artifacts from public/ml.
  const response = await fetch(url)
  // Fail with the HTTP status so the load-check can record a skip.
  if (!response.ok) {
    // Missing export is a load failure, not a silent empty session.
    throw new Error(`Failed to fetch ${url} (${response.status})`)
  }
  // Return the ArrayBuffer InferenceSession.create consumes.
  return response.arrayBuffer()
}

// Dispose a session and ignore backends that lack release().
async function disposeSession(session: ort.InferenceSession | null): Promise<void> {
  // Nothing to do when no session is loaded.
  if (session === null) {
    // Return so callers can always await dispose.
    return
  }
  // InferenceSession.release is the documented ORT Web dispose path.
  const releasable = session as { release?: () => Promise<void> }
  // Older builds may not expose release; skip rather than throw.
  if (typeof releasable.release === 'function') {
    // Free WASM allocations before loading the next heavy graph.
    await releasable.release()
  }
}

// Unload the DistilBERT/LSTM session so the next heavy graph can load.
export async function unloadHeavySession(): Promise<void> {
  // Dispose the active transformer/LSTM session when present.
  if (heavySession !== null) {
    // Release WASM memory before the next checkpoint.
    await disposeSession(heavySession.session)
    // Clear the singleton so a later load does not reuse a released session.
    heavySession = null
  }
}

// Unload the TF-IDF logistic-head session.
export async function unloadTfidfSession(): Promise<void> {
  // Dispose the logistic head when present.
  if (tfidfSession !== null) {
    // Release WASM memory.
    await disposeSession(tfidfSession.session)
    // Clear the singleton.
    tfidfSession = null
  }
}

// Unload every ORT session (used by the sequential load-check between steps).
export async function unloadAllSessions(): Promise<void> {
  // Drop the heavy graph first.
  await unloadHeavySession()
  // Drop the TF-IDF head second.
  await unloadTfidfSession()
}

// Register a newly created DistilBERT or LSTM session, disposing any previous one.
export async function registerHeavySession(
  id: CheckpointId,
  family: CheckpointFamily,
  session: ort.InferenceSession,
): Promise<void> {
  // Never keep two transformer/LSTM graphs in WASM at once.
  await unloadHeavySession()
  // Remember this as the sole heavy session.
  heavySession = { id, family, session }
}

// Register a TF-IDF logistic-head session, disposing a previous TF-IDF graph.
export async function registerTfidfSession(
  id: CheckpointId,
  session: ort.InferenceSession,
): Promise<void> {
  // Replace an older TF-IDF head (10k vs 50k) rather than stacking them.
  await unloadTfidfSession()
  // Remember the logistic-head session.
  tfidfSession = { id, session }
}

// Read the current heavy session, or null when none is loaded.
export function getHeavySession(): { id: CheckpointId; family: CheckpointFamily; session: ort.InferenceSession } | null {
  // Return the singleton without copying the InferenceSession.
  return heavySession
}

// Read the current TF-IDF session, or null when none is loaded.
export function getTfidfSession(): { id: CheckpointId; session: ort.InferenceSession } | null {
  // Return the singleton without copying the InferenceSession.
  return tfidfSession
}

// Re-export Tensor so callers do not import onnxruntime-web directly.
export const OrtTensor = ort.Tensor

// Approximate JS heap in bytes when Chromium exposes performance.memory.
export function readJsHeapBytes(): number | null {
  // performance.memory is a non-standard Chrome extension.
  const memory = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory
  // Return null on Firefox/Safari so the table can show "n/a".
  return memory ? memory.usedJSHeapSize : null
}
