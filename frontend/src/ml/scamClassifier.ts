// Load DistilBERT default, run one inference at a time, and score verified plaintext.
// The chat app does not load TF-IDF or the word BiLSTM. Those graphs stay offline.

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
import { DISTILBERT_OPT_IN_ID } from './types'

// One in-flight DistilBERT default load, shared by StrictMode's double mount.
let distilbertLoad: Promise<void> | null = null
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

// True when the resident session and WordPiece vocab are already DistilBERT default.
export function distilbertDefaultIsResident(): boolean {
  // The heavy singleton must be this graph.
  const heavy = getHeavySession()
  // Vocab, family, and id all have to match or we would score with a stale tokenizer.
  return (
    distilbertVocab?.id === DISTILBERT_OPT_IN_ID &&
    heavy?.id === DISTILBERT_OPT_IN_ID &&
    heavy.family === 'distilbert'
  )
}

// Load DistilBERT default. A resident session returns without fetching again.
export async function loadDistilbertCheckpoint(id: CheckpointId): Promise<CheckpointManifest> {
  // The browser loads one graph. The 512-token export stays offline.
  if (id !== DISTILBERT_OPT_IN_ID) {
    // Refuse so a caller cannot fetch the longer offline graph.
    throw new Error('The browser loads distilbert_default only.')
  }
  // StrictMode remount and a second chat open reuse the session already in WASM.
  if (distilbertDefaultIsResident() && residentManifest !== null) {
    // Do not fetch model.onnx or the WordPiece sidecar again.
    return residentManifest
  }
  // Read the manifest so we know max_length, threshold, filenames, and byte length.
  const manifest = await fetchManifest(id)
  // Guard against a folder that is not DistilBERT.
  if (manifest.family !== 'distilbert') {
    // Fail rather than running the wrong preprocessor.
    throw new Error(`${id} is family ${manifest.family}, not distilbert`)
  }
  // Version both artifacts by the manifest's onnx byte length (no content hash).
  const versionBytes = manifest.onnx_bytes
  // WordPiece filename from the manifest.
  const wordpieceName = manifest.sidecars.wordpiece
  // Read the sidecar from the Cache API, then the network.
  const vocabBytes = await readOrFetchBytes(
    checkpointAssetUrl(id, wordpieceName),
    wordpieceName,
    versionBytes,
  )
  // Decode the cached or fetched JSON. It is a vocabulary, not a message.
  const vocab = JSON.parse(new TextDecoder().decode(vocabBytes)) as WordpieceVocab
  // Read model.onnx the same way. The next visit on this browser hits the cache.
  const onnxBytes = await readOrFetchBytes(
    checkpointAssetUrl(id, manifest.onnx_file),
    manifest.onnx_file,
    versionBytes,
  )
  // Create the WASM session (this is the expensive step).
  const session = await createSessionFromBuffer(onnxBytes)
  // Remember this as the only heavy session.
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

// Load DistilBERT default once. Concurrent callers, including StrictMode, share one promise.
export function ensureDistilbertDefault(): Promise<void> {
  // Already in memory: do not fetch.
  if (distilbertDefaultIsResident()) {
    return Promise.resolve()
  }
  // Share the in-flight load.
  if (distilbertLoad === null) {
    // A failure clears the latch so a later visit can retry. There is no TF-IDF fallback.
    distilbertLoad = loadDistilbertCheckpoint(DISTILBERT_OPT_IN_ID)
      .then(() => undefined)
      .catch((error: unknown) => {
        // Allow a later mount to try again.
        distilbertLoad = null
        // Surface the failure to the preference hook.
        throw error
      })
  }
  // Return the shared promise.
  return distilbertLoad
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
  if (distilbertVocab?.id !== DISTILBERT_OPT_IN_ID) {
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
  // Allow a later ensureDistilbertDefault to start a new fetch.
  distilbertLoad = null
}
