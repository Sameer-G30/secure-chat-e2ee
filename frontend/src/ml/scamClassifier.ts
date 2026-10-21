// Load one DistilBERT-family graph, run one inference at a time, and score verified plaintext.
// Settings chooses distilbert_student (default) or distilbert_best. TF-IDF and the word
// BiLSTM stay offline. At most one heavy ONNX session is resident.

// Import DistilBERT WordPiece helpers (unpadded sequences).
import { buildWordpieceIndex, encodeWordpiece } from './wordpiece'
import type { WordpieceVocab } from './wordpiece'
// Import ORT session helpers. This module does not import tfidf.ts or lstmTokenizer.ts.
import {
  OrtTensor,
  createSessionFromBuffer,
  getHeavySession,
  registerHeavySession,
  unloadAllSessions,
} from './ortRuntime'
// Import the Cache API helper for model.onnx and the WordPiece sidecar.
import { readOrFetchBytes } from './modelCache'
import { checkpointAssetUrl, manifestUrl } from './paths'
import type { CheckpointId, CheckpointManifest, ClassifyResult, FixtureScore } from './types'
import { applyChatScreenThreshold } from './conversationContext'
import { DISTILBERT_BEST_ID, DISTILBERT_FALLBACK_ID, DISTILBERT_OPT_IN_ID } from './types'
import type { BrowserScamModelId } from './types'

// Student, the explicit best opt-in, or the student-missing fallback.
function isServingId(id: CheckpointId | undefined | null): boolean {
  // These three folders share the bert-base WordPiece tokenizer in wordpiece.ts.
  return id === DISTILBERT_OPT_IN_ID || id === DISTILBERT_BEST_ID || id === DISTILBERT_FALLBACK_ID
}

// Bumps when a newer Settings choice starts, so a slow load cannot register after it.
let scamLoadSerial = 0
// One in-flight load for the current choice, shared by StrictMode's double mount.
let inflightModel: { id: BrowserScamModelId; promise: Promise<CheckpointId> } | null = null
// True after the student folder 404s, so a remount can keep the fallback session.
let studentExportMissing = false
// Manifest kept so a second call can return without fetching.
let residentManifest: CheckpointManifest | null = null

// Cached DistilBERT WordPiece vocab for the only browser graph.
let distilbertVocab: {
  id: CheckpointId
  vocab: WordpieceVocab
  tokenToId: Map<string, number>
  threshold: number
} | null = null

// Chain so only one session.run is in flight, even if many messages queue in one tick.
let distilbertQueue: Promise<unknown> = Promise.resolve()
// Chain so two Settings choices cannot compile two graphs at the same time.
let modelLoadQueue: Promise<unknown> = Promise.resolve()

// Run one model load after the previous compile finishes, so peak residency stays one graph.
function runModelLoadExclusive<T>(work: () => Promise<T>): Promise<T> {
  // Attach after the previous load, and keep going if that load failed.
  const run = modelLoadQueue.then(work, work)
  // The chain continues after this load so the next choice can start.
  modelLoadQueue = run.then(
    () => undefined,
    () => undefined,
  )
  // Return this load's resident id to the caller that queued it.
  return run
}

// Run one DistilBERT job after the previous session.run finishes.
function runDistilbertExclusive<T>(work: () => Promise<T>): Promise<T> {
  // Attach after the previous job, and keep going if that job failed.
  const run = distilbertQueue.then(work, work)
  // The chain continues after this job so the next message can start.
  distilbertQueue = run.then(
    () => undefined,
    () => undefined,
  )
  // Return this job's result to the caller that queued it.
  return run
}

// Convert DistilBERT [logit0, logit1] into P(scam) with a stable softmax.
function softmaxScam(logit0: number, logit1: number): number {
  // Subtract max for numerical stability.
  const maxLogit = Math.max(logit0, logit1)
  // Exponentiate the shifted legitimate logit.
  const exp0 = Math.exp(logit0 - maxLogit)
  // Scam class is index 1.
  const exp1 = Math.exp(logit1 - maxLogit)
  // P(scam) = softmax[:, 1].
  return exp1 / (exp0 + exp1)
}

// Fetch JSON from /ml/<id>/<file> and parse it. Used for the small manifest and fixtures.
async function fetchJson<T>(id: CheckpointId, filename: string): Promise<T> {
  // Build the public URL for this sidecar.
  const url = checkpointAssetUrl(id, filename)
  // Fetch the JSON sidecar. The large graph uses the Cache API instead.
  const response = await fetch(url)
  // Fail clearly when the export was not copied into public/ml.
  if (!response.ok) {
    // Include the URL so an operator can see which artifact is missing.
    throw new Error(`Failed to fetch ${url} (${response.status})`)
  }
  // Parse the JSON body.
  return (await response.json()) as T
}

// Fetch and parse a checkpoint manifest.
export async function fetchManifest(id: CheckpointId): Promise<CheckpointManifest> {
  // Manifest lives at /ml/<id>/manifest.json.
  const response = await fetch(manifestUrl(id))
  // Missing manifest means the exporter has not been run.
  if (!response.ok) {
    // Tell the caller to record a skip rather than crashing the tab.
    throw new Error(`Failed to fetch manifest for ${id} (${response.status})`)
  }
  // Parse the catalog metadata.
  return (await response.json()) as CheckpointManifest
}

// True when some serving DistilBERT session and its WordPiece vocab are resident.
export function distilbertDefaultIsResident(): boolean {
  // The heavy singleton must be this graph.
  const heavy = getHeavySession()
  // Vocab, family, and id all have to match or we would score with a stale tokenizer.
  return (
    isServingId(distilbertVocab?.id) &&
    heavy !== null &&
    heavy.id === distilbertVocab?.id &&
    heavy.family === 'distilbert'
  )
}

// Release a session that lost the race with a newer Settings choice.
async function releaseOrphanSession(session: { release?: () => Promise<void> }): Promise<void> {
  // InferenceSession.release frees the WASM graph.
  if (typeof session.release === 'function') {
    // Drop the graph that must not stay beside the chosen one.
    await session.release()
  }
}

// Load one serving DistilBERT graph. A resident session of the same id returns without fetching.
export async function loadDistilbertCheckpoint(
  id: CheckpointId,
  serial?: number,
): Promise<CheckpointManifest> {
  // Chat loads the student, the best opt-in, or the student-missing fallback. Not TF-IDF or LSTM.
  if (!isServingId(id)) {
    // Refuse so a caller cannot fetch an offline graph.
    throw new Error('The browser loads distilbert_student, distilbert_best, or distilbert_default.')
  }
  // Undefined serial means the caller is the load-check page, which is the only load in flight.
  const stillCurrent = (): boolean => serial === undefined || serial === scamLoadSerial
  // Reuse this exact graph. A different resident id must be unloaded first.
  if (distilbertDefaultIsResident() && residentManifest !== null && residentManifest.id === id) {
    // Do not fetch model.onnx or the WordPiece sidecar again.
    return residentManifest
  }
  // A newer choice started while we were queued.
  if (!stillCurrent()) {
    // Do not fetch the graph the user no longer wants.
    throw new Error('Scam model load was replaced by a newer choice.')
  }
  // Unload any other heavy graph before InferenceSession.create, so two graphs are not resident.
  if (getHeavySession() !== null) {
    // reset clears vocab and the WASM session before the new bytes are parsed.
    await resetAllClassifiers()
  }
  // The choice may have changed during dispose.
  if (!stillCurrent()) {
    // Leave memory empty for the newer load.
    throw new Error('Scam model load was replaced by a newer choice.')
  }
  // Read the manifest so we know max_length, threshold, filenames, and byte length.
  const manifest = await fetchManifest(id)
  // Guard against a folder that is not DistilBERT.
  if (manifest.family !== 'distilbert') {
    // Fail rather than running the wrong preprocessor.
    throw new Error(`${id} is family ${manifest.family}, not distilbert`)
  }
  // Version by content hash when the manifest has one, else by byte length.
  const versionBytes: number | string = manifest.sha256 ?? manifest.onnx_bytes
  // WordPiece filename from the manifest.
  const wordpieceName = manifest.sidecars.wordpiece
  // Read the sidecar from the Cache API, then the network.
  const vocabBytes = await readOrFetchBytes(
    checkpointAssetUrl(id, wordpieceName),
    wordpieceName,
    versionBytes,
    id,
  )
  // Decode the cached or fetched JSON. It is a vocabulary, not a message.
  const vocab = JSON.parse(new TextDecoder().decode(vocabBytes)) as WordpieceVocab
  // Read model.onnx the same way. The next visit on this browser hits the cache.
  const onnxBytes = await readOrFetchBytes(
    checkpointAssetUrl(id, manifest.onnx_file),
    manifest.onnx_file,
    versionBytes,
    id,
  )
  // Create the WASM session (this is the expensive step). The previous graph is already unloaded.
  const session = await createSessionFromBuffer(onnxBytes)
  // A newer choice won while this graph was compiling. Drop it before register.
  if (!stillCurrent()) {
    // release so this orphan does not sit beside the chosen session.
    await releaseOrphanSession(session)
    // Tell the caller this load is stale.
    throw new Error('Scam model load was replaced by a newer choice.')
  }
  // Remember this as the only heavy session. registerHeavySession unloads again if one slipped in.
  await registerHeavySession(id, 'distilbert', session)
  // Build the token → id map once.
  const tokenToId = buildWordpieceIndex(vocab)
  // Cache vocab for classifyDistilbert.
  distilbertVocab = { id, vocab, tokenToId, threshold: manifest.threshold }
  // Remember the manifest so the next call can skip the network.
  residentManifest = manifest
  // Return the manifest.
  return manifest
}

// Load the Settings choice. Student falls back to distilbert_default when its export is missing.
// Best does not fall back. A failure leaves no heavy session resident.
async function loadRequestedModel(requested: BrowserScamModelId, serial: number): Promise<CheckpointId> {
  // Drop the previous graph before creating the next one.
  await resetAllClassifiers()
  // A newer click happened during dispose.
  if (serial !== scamLoadSerial) {
    // The newer call owns the load.
    throw new Error('Scam model load was replaced by a newer choice.')
  }
  // The default choice is the student, with the historical fallback if that folder is absent.
  if (requested === DISTILBERT_OPT_IN_ID) {
    try {
      // Prefer the 17 MiB student.
      const manifest = await loadDistilbertCheckpoint(DISTILBERT_OPT_IN_ID, serial)
      // The student folder exists, so a later remount should not skip it.
      studentExportMissing = false
      // The resident id is the student.
      return manifest.id
    } catch (error) {
      // A superseded load must not start the 64 MiB fallback on top of the new choice.
      if (serial !== scamLoadSerial) {
        // Surface the cancellation.
        throw error
      }
      // Remember the miss so StrictMode can keep the fallback without a second 404.
      studentExportMissing = true
      // Fall back to distilbert_default. This replaces nothing, because the student never registered.
      const manifest = await loadDistilbertCheckpoint(DISTILBERT_FALLBACK_ID, serial)
      // Cache and scoring use the graph that actually loaded.
      return manifest.id
    }
  }
  // DistilBERT best is explicit. A missing export leaves banners off rather than loading a second graph.
  const manifest = await loadDistilbertCheckpoint(requested, serial)
  // Resident id is distilbert_best.
  return manifest.id
}

// Load the chosen graph, unloading any other heavy session first. Returns the resident id.
export function ensureScamModel(requested: BrowserScamModelId): Promise<CheckpointId> {
  // The id of the graph whose vocab and session are both live.
  const resident = distilbertDefaultIsResident() ? (distilbertVocab?.id ?? null) : null
  // Same choice already resident: do not fetch or compile again.
  if (resident === requested) {
    // Callers still need the id for the banner cache.
    return Promise.resolve(requested)
  }
  // The student file was missing and the fallback is already the only session.
  if (requested === DISTILBERT_OPT_IN_ID && studentExportMissing && resident === DISTILBERT_FALLBACK_ID) {
    // Keep that one graph. Do not load the student and the fallback together.
    return Promise.resolve(DISTILBERT_FALLBACK_ID)
  }
  // StrictMode remount shares the load already in flight for this choice.
  if (inflightModel?.id === requested) {
    // One compile for both mounts.
    return inflightModel.promise
  }
  // This click invalidates any older in-flight compile.
  const serial = ++scamLoadSerial
  // Queue behind any compile already in flight so the two creates cannot overlap.
  const promise = runModelLoadExclusive(() => loadRequestedModel(requested, serial)).finally(() => {
    // Only the promise we published may clear the latch.
    if (inflightModel?.promise === promise) {
      // A later choice can start its own load.
      inflightModel = null
    }
  })
  // Publish before the first await so a second mount can share it.
  inflightModel = { id: requested, promise }
  // Return the resident id, or reject when the graph cannot load.
  return promise
}

// Load the default student once. Concurrent callers, including StrictMode, share one promise.
export function ensureDistilbertDefault(): Promise<void> {
  // The chat default is the student. Best is only loaded from Settings.
  return ensureScamModel(DISTILBERT_OPT_IN_ID).then(() => undefined)
}

// Run the loaded DistilBERT graph on one string. Calls are serialized on the ORT worker.
export async function classifyDistilbert(text: string): Promise<ClassifyResult> {
  // Queue behind any session.run that is already in flight.
  return runDistilbertExclusive(() => classifyDistilbertNow(text))
}

// Tokenize and run one DistilBERT forward pass. Only called from the exclusive queue.
async function classifyDistilbertNow(text: string): Promise<ClassifyResult> {
  // Require a loaded WordPiece vocab.
  const loaded = distilbertVocab
  // The ORT session is the heavy singleton.
  const sessionHolder = getHeavySession()
  // Both must be present, DistilBERT family, same id.
  if (
    !loaded ||
    !sessionHolder ||
    sessionHolder.family !== 'distilbert' ||
    sessionHolder.id !== loaded.id
  ) {
    // Fail rather than scoring with a stale tokenizer.
    throw new Error('DistilBERT checkpoint is not loaded')
  }
  // Time tokenize + MatMul path.
  const started = performance.now()
  // WordPiece encode with truncation only (no pad-to-max_length).
  const encoded = encodeWordpiece(text, loaded.vocab, loaded.tokenToId)
  // ORT wants int64 input_ids.
  const inputIds = new OrtTensor(
    'int64',
    BigInt64Array.from(encoded.inputIds.map((tokenId) => BigInt(tokenId))),
    [1, encoded.inputIds.length],
  )
  // ORT wants int64 attention_mask.
  const attentionMask = new OrtTensor(
    'int64',
    BigInt64Array.from(encoded.attentionMask.map((bit) => BigInt(bit))),
    [1, encoded.attentionMask.length],
  )
  // Run DistilBERT. The exclusive queue means this is the only run in flight.
  const output = await sessionHolder.session.run({
    input_ids: inputIds,
    attention_mask: attentionMask,
  })
  // Logits are [1, 2].
  const logits = output.logits
  // Read the legitimate logit.
  const logit0 = Number(logits.data[0])
  // Scam logit is index 1.
  const logit1 = Number(logits.data[1])
  // Convert to P(scam).
  const pScam = softmaxScam(logit0, logit1)
  // Stop the timer.
  const inferenceMs = performance.now() - started
  // Apply the VAL-frozen threshold from the manifest (0.30). ChatScreen then applies 0.35.
  const warned = pScam >= loaded.threshold
  // Return the unified classify result.
  return { pScam, warned, checkpointId: loaded.id, inferenceMs }
}

// Fetch fixture_scores.json written at export time (Python/PyTorch reference).
export async function fetchFixtureScores(id: CheckpointId, filename: string): Promise<FixtureScore[]> {
  // Parse the fixture list used by the DistilBERT-only load check.
  return fetchJson<FixtureScore[]>(id, filename)
}

// Score verified plaintext with DistilBERT default only. Null when that graph is not loaded.
export async function classifyVerifiedPlaintext(text: string): Promise<ClassifyResult | null> {
  // Do not score until distilbert_default is the resident session.
  if (!isServingId(distilbertVocab?.id)) {
    // The caller leaves banners off. There is no TF-IDF or LSTM fallback.
    return null
  }
  try {
    // Run the transformer, then apply the ChatScreen 0.35 overlay.
    return applyChatScreenThreshold(await classifyDistilbert(text))
  } catch {
    // WASM abort: skip the banner rather than falling through to another model.
    return null
  }
}

// Unload the DistilBERT session (the load-check page resets before and after its one step).
export async function resetAllClassifiers(): Promise<void> {
  // Dispose the WASM session.
  await unloadAllSessions()
  // Drop the WordPiece cache so the next load cannot mix vocabs.
  distilbertVocab = null
  // Drop the manifest so the next load fetches again.
  residentManifest = null
  // A full reset retries the student folder instead of pinning the fallback.
  studentExportMissing = false
}
