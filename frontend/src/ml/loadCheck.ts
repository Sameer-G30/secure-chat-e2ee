// Sequential ONNX Runtime Web load-and-check of all six Slice-6 checkpoints.

// Import the family loaders and fixture scorer.
import {
  classifyDistilbert,
  classifyLstm,
  classifyTfidf,
  fetchFixtureScores,
  fetchManifest,
  loadDistilbertCheckpoint,
  loadLstmCheckpoint,
  loadTfidfCheckpoint,
  resetAllClassifiers,
} from './scamClassifier'
import { readJsHeapBytes } from './ortRuntime'
import type { CheckpointId, LoadCheckRow } from './types'
import { CHECKPOINT_LOAD_ORDER } from './types'

// React StrictMode mounts this page twice; share one WASM sequence across remounts.
let sharedRun: Promise<LoadCheckRow[]> | null = null
// Rows already finished, replayed when a remounted screen subscribes.
const completedRows: LoadCheckRow[] = []
// Live UI listeners; remount replaces the cancelled StrictMode callback.
const rowListeners = new Set<(row: LoadCheckRow, index: number) => void>()

// Load one checkpoint, score fixtures, then unload before the next id.
async function checkOne(id: CheckpointId, loadOrder: number): Promise<LoadCheckRow> {
  // Read the manifest first so a missing export is recorded as load failure.
  // Human-readable label starts as the catalog id until the manifest loads.
  let manifestLabel: string = id
  // Offline citation is filled when the manifest fetch succeeds.
  let offline: LoadCheckRow['offline'] = null
  // ONNX byte size from the manifest.
  let onnxBytes: number | null = null
  try {
    // Fetch catalog metadata without creating a WASM session yet.
    const manifest = await fetchManifest(id)
    // Copy the human-readable label into the table row.
    manifestLabel = manifest.label
    // Copy offline TEST/chat-eval numbers (not browser accuracy).
    offline = manifest.offline
    // Copy serving graph size.
    onnxBytes = manifest.onnx_bytes
    // Time InferenceSession.create + sidecar JSON parses.
    const initStarted = performance.now()
    // Dispatch to the family loader (each unloads the previous heavy graph).
    if (manifest.family === 'distilbert') {
      // DistilBERT winner (512) or Slice 5 default (256).
      await loadDistilbertCheckpoint(id)
    } else if (manifest.family === 'lstm') {
      // Word BiLSTM winner (8 ep) or published default (4 ep).
      await loadLstmCheckpoint(id)
    } else {
      // TF-IDF winner (10k) or published default (50k).
      await loadTfidfCheckpoint(id)
    }
    // Session create latency (load succeeded even if a later fixture throws).
    const initMs = performance.now() - initStarted
    // Approximate JS heap after load (Chrome only).
    const jsHeapBytes = readJsHeapBytes()
    try {
      // Load Python/PyTorch fixture expectations written at export time.
      const fixtures = await fetchFixtureScores(id, manifest.sidecars.fixtures)
      // Accumulate per-message inference times.
      let inferenceTotal = 0
      // Record whether the browser banner matches the Python banner.
      const bannerMatch: boolean[] = []
      // Score each short DM with the live ORT session.
      for (const fixture of fixtures) {
        // Time this one message.
        const messageStarted = performance.now()
        // Dispatch to the family classifier.
        const result =
          manifest.family === 'distilbert'
            ? await classifyDistilbert(fixture.text)
            : manifest.family === 'lstm'
              ? await classifyLstm(fixture.text)
              : await classifyTfidf(fixture.text)
        // Add wall-clock inference (includes tokenize in JS).
        inferenceTotal += performance.now() - messageStarted
        // Compare banner on/off only; int8 DistilBERT may drift in P(scam).
        bannerMatch.push(result.warned === fixture.warned)
      }
      // Mean milliseconds per fixture DM.
      const inferenceMsPerMessage = fixtures.length > 0 ? inferenceTotal / fixtures.length : null
      // Unload this graph before the next checkpoint (one heavy session at a time).
      await resetAllClassifiers()
      // Return a successful row.
      return {
        id,
        loadOrder,
        label: manifestLabel,
        loadSuccess: true,
        error: null,
        initMs,
        inferenceMsPerMessage,
        onnxBytes,
        jsHeapBytes,
        fixtureBannerMatch: bannerMatch,
        offline,
      }
    } catch (inferError) {
      // Session.create worked; record inference separately so a classify bug is not an OOM.
      const inferMessage = inferError instanceof Error ? inferError.message : String(inferError)
      // Drop the session so the next catalog id starts clean.
      await resetAllClassifiers().catch(() => undefined)
      // loadSuccess stays true: the graph entered WASM.
      return {
        id,
        loadOrder,
        label: manifestLabel,
        loadSuccess: true,
        error: `inference: ${inferMessage}`,
        initMs,
        inferenceMsPerMessage: null,
        onnxBytes,
        jsHeapBytes,
        fixtureBannerMatch: null,
        offline,
      }
    }
  } catch (error) {
    // Record the failure and continue the six-way sequence.
    const message = error instanceof Error ? error.message : String(error)
    // Make sure a crashed session does not leak into the next step.
    await resetAllClassifiers().catch(() => undefined)
    // Return a failed row; the caller still proceeds to the next id.
    return {
      id,
      loadOrder,
      label: manifestLabel,
      loadSuccess: false,
      error: message,
      initMs: null,
      inferenceMsPerMessage: null,
      onnxBytes,
      jsHeapBytes: readJsHeapBytes(),
      fixtureBannerMatch: null,
      offline,
    }
  }
}

// Walk checkpoints 1..6 once; StrictMode remounts subscribe instead of starting a second WASM run.
async function runSequentialLoadCheckOnce(
  emit: (row: LoadCheckRow, index: number) => void,
): Promise<LoadCheckRow[]> {
  // Start from a clean WASM heap.
  await resetAllClassifiers()
  // Walk the documented order (DistilBERT 512 first).
  for (let index = 0; index < CHECKPOINT_LOAD_ORDER.length; index += 1) {
    // Catalog id for this step.
    const id = CHECKPOINT_LOAD_ORDER[index]
    // Skip holes (should never happen).
    if (!id) {
      // Continue so a typo cannot abort the remaining checks.
      continue
    }
    // Load, score fixtures, unload.
    const row = await checkOne(id, index + 1)
    // Remember the row so a remounted UI can replay it.
    completedRows.push(row)
    // Notify every live subscriber (current Chat-less measurement page).
    emit(row, index)
  }
  // Return the completed six-row log.
  return completedRows
}

// Run checkpoints 1..6 in order; a failure skips to the next id.
export async function runSequentialLoadCheck(
  onRow?: (row: LoadCheckRow, index: number) => void,
): Promise<LoadCheckRow[]> {
  // Replay rows that finished before this subscriber attached (StrictMode remount).
  if (onRow) {
    // Register before replay so a row finishing mid-replay cannot be dropped.
    rowListeners.add(onRow)
    // Replay already-completed rows into the new React tree.
    completedRows.forEach((row, index) => {
      // Index matches loadOrder - 1 for the documented catalog.
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
    sharedRun = runSequentialLoadCheckOnce(emit)
  }
  try {
    // Await the shared six-way sequence.
    return await sharedRun
  } finally {
    // Drop this instance's callback so an unmounted tree cannot setState.
    if (onRow) {
      // Unsubscribe after the promise settles or the caller unmounts via .finally below.
      rowListeners.delete(onRow)
    }
  }
}

// Remove a UI callback when the measurement page unmounts mid-run.
export function unsubscribeLoadCheck(onRow: (row: LoadCheckRow, index: number) => void): void {
  // Ignore unknown callbacks so cleanup is always safe.
  rowListeners.delete(onRow)
}
