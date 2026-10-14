// On-device load check for DistilBERT default only.
// TF-IDF and the word BiLSTM are not loaded here. The six-row table in
// ml/reports/onnx_web_load_check.md stays a historical measurement.

// Import the DistilBERT loader and fixture scorer. This file does not import tfidf or lstm.
import {
  classifyDistilbert,
  fetchFixtureScores,
  fetchManifest,
  loadDistilbertCheckpoint,
  resetAllClassifiers,
} from './scamClassifier'
import { readJsHeapBytes } from './ortRuntime'
import { percentile } from './percentiles'
import type { LoadCheckRow } from './types'
import { DISTILBERT_OPT_IN_ID } from './types'

// React StrictMode mounts this page twice; share one WASM sequence across remounts.
let sharedRun: Promise<LoadCheckRow[]> | null = null
// Rows already finished, replayed when a remounted screen subscribes.
const completedRows: LoadCheckRow[] = []
// Live UI listeners; remount replaces the cancelled StrictMode callback.
const rowListeners = new Set<(row: LoadCheckRow, index: number) => void>()

// Load DistilBERT default, score its fixtures, then unload.
async function checkDistilbertDefault(): Promise<LoadCheckRow> {
  // Human-readable label starts as the catalog id until the manifest loads.
  let manifestLabel: string = DISTILBERT_OPT_IN_ID
  // Offline citation is filled when the manifest fetch succeeds.
  let offline: LoadCheckRow['offline'] = null
  // ONNX byte size from the manifest.
  let onnxBytes: number | null = null
  try {
    // Fetch catalog metadata without creating a WASM session yet.
    const manifest = await fetchManifest(DISTILBERT_OPT_IN_ID)
    // Copy the human-readable label into the table row.
    manifestLabel = manifest.label
    // Copy offline TEST/chat-eval numbers (not browser accuracy).
    offline = manifest.offline
    // Copy serving graph size.
    onnxBytes = manifest.onnx_bytes
    // Time InferenceSession.create + sidecar reads.
    const initStarted = performance.now()
    // Load the only browser graph.
    await loadDistilbertCheckpoint(DISTILBERT_OPT_IN_ID)
    // Session create latency.
    const initMs = performance.now() - initStarted
    // Approximate JS heap after load (Chrome only).
    const jsHeapBytes = readJsHeapBytes()
    try {
      // Load Python/PyTorch fixture expectations written at export time.
      const fixtures = await fetchFixtureScores(DISTILBERT_OPT_IN_ID, manifest.sidecars.fixtures)
      // Accumulate per-message inference times.
      const inferenceSamples: number[] = []
      // Record whether the browser banner matches the Python banner.
      const bannerMatch: boolean[] = []
      // Score each short DM with the live ORT session, one at a time.
      for (const fixture of fixtures) {
        // Time this one message.
        const messageStarted = performance.now()
        // DistilBERT only. The sidecar threshold is used here, not the ChatScreen 0.35 overlay.
        const result = await classifyDistilbert(fixture.text)
        // Add wall-clock inference (includes tokenize in JS).
        inferenceSamples.push(performance.now() - messageStarted)
        // Compare banner on/off only.
        bannerMatch.push(result.warned === fixture.warned)
      }
      // Mean milliseconds per fixture DM.
      const inferenceMsPerMessage =
        inferenceSamples.length > 0
          ? inferenceSamples.reduce((sum, value) => sum + value, 0) / inferenceSamples.length
          : null
      // Percentiles over the same fixtures.
      const inferenceMsP50 = percentile(inferenceSamples, 50)
      const inferenceMsP95 = percentile(inferenceSamples, 95)
      const inferenceMsP99 = percentile(inferenceSamples, 99)
      // Unload so a later chat mount starts from a known heap.
      await resetAllClassifiers()
      // Return a successful row.
      return {
        id: DISTILBERT_OPT_IN_ID,
        loadOrder: 1,
        label: manifestLabel,
        loadSuccess: true,
        error: null,
        initMs,
        inferenceMsPerMessage,
        inferenceMsP50,
        inferenceMsP95,
        inferenceMsP99,
        onnxBytes,
        jsHeapBytes,
        fixtureBannerMatch: bannerMatch,
        offline,
      }
    } catch (inferError) {
      // Session.create worked; record inference separately.
      const inferMessage = inferError instanceof Error ? inferError.message : String(inferError)
      // Drop the session.
      await resetAllClassifiers().catch(() => undefined)
      // loadSuccess stays true: the graph entered WASM.
      return {
        id: DISTILBERT_OPT_IN_ID,
        loadOrder: 1,
        label: manifestLabel,
        loadSuccess: true,
        error: `inference: ${inferMessage}`,
        initMs,
        inferenceMsPerMessage: null,
        inferenceMsP50: null,
        inferenceMsP95: null,
        inferenceMsP99: null,
        onnxBytes,
        jsHeapBytes,
        fixtureBannerMatch: null,
        offline,
      }
    }
  } catch (error) {
    // Record the failure. Do not try another family.
    const message = error instanceof Error ? error.message : String(error)
    // Make sure a crashed session does not leak.
    await resetAllClassifiers().catch(() => undefined)
    // Return a failed row.
    return {
      id: DISTILBERT_OPT_IN_ID,
      loadOrder: 1,
      label: manifestLabel,
      loadSuccess: false,
      error: message,
      initMs: null,
      inferenceMsPerMessage: null,
      inferenceMsP50: null,
      inferenceMsP95: null,
      inferenceMsP99: null,
      onnxBytes,
      jsHeapBytes: readJsHeapBytes(),
      fixtureBannerMatch: null,
      offline,
    }
  }
}

// Run DistilBERT default once; StrictMode remounts subscribe instead of starting a second load.
async function runOnce(emit: (row: LoadCheckRow, index: number) => void): Promise<LoadCheckRow[]> {
  // Start from a clean WASM heap.
  await resetAllClassifiers()
  // One graph. TF-IDF and LSTM URLs are not requested.
  const row = await checkDistilbertDefault()
  // Remember the row so a remounted UI can replay it.
  completedRows.push(row)
  // Notify every live subscriber.
  emit(row, 0)
  // Return the one-row log.
  return completedRows
}

// Run the DistilBERT-only check.
export async function runSequentialLoadCheck(
  onRow?: (row: LoadCheckRow, index: number) => void,
): Promise<LoadCheckRow[]> {
  // Replay rows that finished before this subscriber attached.
  if (onRow) {
    // Register before replay so a row finishing mid-replay cannot be dropped.
    rowListeners.add(onRow)
    // Replay already-completed rows into the new React tree.
    completedRows.forEach((row, index) => {
      // Index is 0 for the single DistilBERT default row.
      onRow(row, index)
    })
  }
  // Fan-out helper used by the singleton runner.
  const emit = (row: LoadCheckRow, index: number) => {
    // Copy listeners in case a subscriber unregisters during emit.
    for (const listener of [...rowListeners]) {
      // Each measurement page instance updates its own table.
      listener(row, index)
    }
  }
  // Start the WASM sequence at most once per page lifetime.
  if (sharedRun === null) {
    // Assign immediately so a synchronous second call cannot start another loop.
    sharedRun = runOnce(emit)
  }
  try {
    // Await the shared run.
    return await sharedRun
  } finally {
    // Drop this instance's callback so an unmounted tree cannot setState.
    if (onRow) {
      rowListeners.delete(onRow)
    }
  }
}

// Remove a UI callback when the measurement page unmounts mid-run.
export function unsubscribeLoadCheck(onRow: (row: LoadCheckRow, index: number) => void): void {
  // Ignore unknown callbacks so cleanup is always safe.
  rowListeners.delete(onRow)
}
