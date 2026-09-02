// Load one ONNX checkpoint, run inference on verified plaintext, unload on demand.

// Import URL scaling used by TF-IDF and the word BiLSTM.
import { scaleUrlFeatures, featuresToVector, extractMessageUrlFeatures } from './urlFeatures'
import type { UrlScalerSidecar } from './urlFeatures'
// Import the TypeScript TF-IDF vectorizer (A5).
import { buildTfidfVocabularyIndex, vectorizeTfidfUnionIndexed } from './tfidf'
import type { TfidfSidecar } from './tfidf'
// Import DistilBERT WordPiece helpers (A6).
import { buildWordpieceIndex, encodeWordpiece } from './wordpiece'
import type { WordpieceVocab } from './wordpiece'
// Import the word-BiLSTM tokenizer (ChatScreen lazy-loads lstm_best behind a toggle).
import { encodeLstmUnpadded } from './lstmTokenizer'
import type { LstmMetaSidecar } from './lstmTokenizer'
// Import ORT session helpers that enforce one heavy graph at a time.
import {
  OrtTensor,
  createSessionFromBuffer,
  fetchOnnxBuffer,
  getHeavySession,
  getTfidfSession,
  registerHeavySession,
  registerTfidfSession,
  unloadAllSessions,
  unloadHeavySession,
} from './ortRuntime'
import { checkpointAssetUrl, manifestUrl } from './paths'
import type {
  ChatHeavyPreference,
  CheckpointId,
  CheckpointManifest,
  ClassifyResult,
  FixtureScore,
} from './types'
import { CHATSCREEN_DEFAULT_ID, DISTILBERT_OPT_IN_ID, LSTM_OPT_IN_ID } from './types'

// Convert DistilBERT/LSTM [logit0, logit1] into P(scam) with a stable softmax.
function softmaxScam(logit0: number, logit1: number): number {
  // Subtract max for numerical stability.
  const maxLogit = Math.max(logit0, logit1)
  // Exponentiate the shifted logits.
  const exp0 = Math.exp(logit0 - maxLogit)
  // Scam class is index 1.
  const exp1 = Math.exp(logit1 - maxLogit)
  // P(scam) = softmax[:, 1].
  return exp1 / (exp0 + exp1)
}

// Fetch JSON from /ml/<id>/<file> and parse it.
async function fetchJson<T>(id: CheckpointId, filename: string): Promise<T> {
  // Build the public URL for this sidecar.
  const url = checkpointAssetUrl(id, filename)
  // Fetch the JSON sidecar.
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
    // Tell the load-check to record a skip rather than crashing the tab.
    throw new Error(`Failed to fetch manifest for ${id} (${response.status})`)
  }
  // Parse the catalog metadata.
  return (await response.json()) as CheckpointManifest
}

// Cached TF-IDF sidecars so ChatScreen does not re-fetch 50k terms every message.
let tfidfSidecar: { id: CheckpointId; sidecar: TfidfSidecar; scaler: UrlScalerSidecar; vocabIndex: Map<string, number> } | null =
  null

// Cached DistilBERT WordPiece vocab for the opt-in graph.
let distilbertVocab: {
  id: CheckpointId
  vocab: WordpieceVocab
  tokenToId: Map<string, number>
  threshold: number
} | null = null

// Cached LSTM meta for the sequential check (not used by ChatScreen).
let lstmMeta: { id: CheckpointId; meta: LstmMetaSidecar; scaler: UrlScalerSidecar } | null = null

// Load TF-IDF JSON sidecars and the logistic-head ONNX graph.
export async function loadTfidfCheckpoint(id: CheckpointId): Promise<CheckpointManifest> {
  // Read the manifest so we know filenames and the frozen threshold.
  const manifest = await fetchManifest(id)
  // Guard against loading a DistilBERT folder as TF-IDF.
  if (manifest.family !== 'tfidf') {
    // Fail rather than running the wrong preprocessor.
    throw new Error(`${id} is family ${manifest.family}, not tfidf`)
  }
  // Vocabulary + idf + threshold.
  const sidecar = await fetchJson<TfidfSidecar>(id, manifest.sidecars.tfidf)
  // TRAIN-fitted URL scaler.
  const scaler = await fetchJson<UrlScalerSidecar>(id, manifest.sidecars.url_scaler)
  // Fetch the tiny Gemm+Sigmoid graph.
  const buffer = await fetchOnnxBuffer(id, manifest.onnx_file)
  // Create the WASM session.
  const session = await createSessionFromBuffer(buffer)
  // Remember this as the TF-IDF session (replacing 10k vs 50k).
  await registerTfidfSession(id, session)
  // Build the O(1) term → column map once.
  const vocabIndex = buildTfidfVocabularyIndex(sidecar)
  // Cache sidecars for classifyTfidf.
  tfidfSidecar = { id, sidecar, scaler, vocabIndex }
  // Return the manifest so the load-check can cite offline metrics.
  return manifest
}

// Load DistilBERT ONNX + WordPiece vocab; unloads any previous heavy graph.
export async function loadDistilbertCheckpoint(id: CheckpointId): Promise<CheckpointManifest> {
  // Read the manifest so we know max_length, threshold, and filenames.
  const manifest = await fetchManifest(id)
  // Guard against loading a TF-IDF folder as DistilBERT.
  if (manifest.family !== 'distilbert') {
    // Fail rather than running the wrong preprocessor.
    throw new Error(`${id} is family ${manifest.family}, not distilbert`)
  }
  // WordPiece token list.
  const vocab = await fetchJson<WordpieceVocab>(id, manifest.sidecars.wordpiece)
  // Fetch the (preferably int8) DistilBERT graph.
  const buffer = await fetchOnnxBuffer(id, manifest.onnx_file)
  // Create the WASM session (this is the expensive step).
  const session = await createSessionFromBuffer(buffer)
  // Dispose any LSTM/other DistilBERT session first.
  await registerHeavySession(id, 'distilbert', session)
  // DistilBERT replaced any LSTM graph; drop the LSTM tokenizer cache.
  lstmMeta = null
  // Build the token → id map once.
  const tokenToId = buildWordpieceIndex(vocab)
  // Cache vocab for classifyDistilbert.
  distilbertVocab = { id, vocab, tokenToId, threshold: manifest.threshold }
  // Return the manifest.
  return manifest
}

// Load word-BiLSTM ONNX + vocab/scaler; unloads any previous heavy graph.
export async function loadLstmCheckpoint(id: CheckpointId): Promise<CheckpointManifest> {
  // Read the manifest so we know filenames and the frozen threshold.
  const manifest = await fetchManifest(id)
  // Guard against loading the wrong family.
  if (manifest.family !== 'lstm') {
    // Fail rather than running the wrong preprocessor.
    throw new Error(`${id} is family ${manifest.family}, not lstm`)
  }
  // Vocab + scaler + threshold.
  const meta = await fetchJson<LstmMetaSidecar>(id, manifest.sidecars.lstm_meta)
  // Shared URL scaler sidecar (same 20-d layout).
  const scaler = await fetchJson<UrlScalerSidecar>(id, manifest.sidecars.url_scaler)
  // Fetch the fp32 LSTM graph.
  const buffer = await fetchOnnxBuffer(id, manifest.onnx_file)
  // Create the WASM session.
  const session = await createSessionFromBuffer(buffer)
  // Dispose any DistilBERT session first (one heavy graph at a time).
  await registerHeavySession(id, 'lstm', session)
  // LSTM replaced any DistilBERT graph; drop the WordPiece cache.
  distilbertVocab = null
  // Cache meta for classifyLstm.
  lstmMeta = { id, meta, scaler }
  // Return the manifest.
  return manifest
}

// Run the loaded TF-IDF logistic head on one DM.
export async function classifyTfidf(text: string): Promise<ClassifyResult> {
  // Require a loaded TF-IDF session and sidecar.
  const loaded = tfidfSidecar
  // The ORT session is stored separately from the sidecar cache.
  const sessionHolder = getTfidfSession()
  // Both must be present and refer to the same checkpoint.
  if (!loaded || !sessionHolder || sessionHolder.id !== loaded.id) {
    // Fail rather than scoring with a stale vocab.
    throw new Error('TF-IDF checkpoint is not loaded')
  }
  // Time the vectorize + Gemm path.
  const started = performance.now()
  // Build the FeatureUnion dense row in TypeScript.
  const features = vectorizeTfidfUnionIndexed(
    text,
    loaded.sidecar,
    loaded.scaler,
    loaded.vocabIndex,
  )
  // Wrap as a float32 tensor [1, n_features].
  const input = new OrtTensor('float32', features, [1, features.length])
  // Run Gemm+Sigmoid.
  const output = await sessionHolder.session.run({ features: input })
  // The graph's output name is 'probabilities'.
  const probs = output.probabilities
  // Read P(scam) from the first (only) batch row.
  const pScam = Number(probs.data[0])
  // Stop the timer after ORT returns.
  const inferenceMs = performance.now() - started
  // Apply the VAL-frozen threshold.
  const warned = pScam >= loaded.sidecar.threshold
  // Return the unified classify result.
  return { pScam, warned, checkpointId: loaded.id, inferenceMs }
}

// Run the loaded DistilBERT graph on one DM.
export async function classifyDistilbert(text: string): Promise<ClassifyResult> {
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
  // WordPiece encode with truncation only (no pad-to-max_length; dynamic sequence axis).
  const encoded = encodeWordpiece(text, loaded.vocab, loaded.tokenToId)
  // ORT wants int64 input_ids.
  const inputIds = new OrtTensor(
    'int64',
    BigInt64Array.from(encoded.inputIds.map((id) => BigInt(id))),
    [1, encoded.inputIds.length],
  )
  // ORT wants int64 attention_mask.
  const attentionMask = new OrtTensor(
    'int64',
    BigInt64Array.from(encoded.attentionMask.map((bit) => BigInt(bit))),
    [1, encoded.attentionMask.length],
  )
  // Run DistilBERT.
  const output = await sessionHolder.session.run({
    input_ids: inputIds,
    attention_mask: attentionMask,
  })
  // Logits are [1, 2].
  const logits = output.logits
  // Read the two class logits.
  const logit0 = Number(logits.data[0])
  // Scam logit is index 1.
  const logit1 = Number(logits.data[1])
  // Convert to P(scam).
  const pScam = softmaxScam(logit0, logit1)
  // Stop the timer.
  const inferenceMs = performance.now() - started
  // Apply the VAL-frozen threshold from the manifest (never retuned on chat_eval).
  const warned = pScam >= loaded.threshold
  // Return the unified classify result.
  return { pScam, warned, checkpointId: loaded.id, inferenceMs }
}

// Run the loaded word BiLSTM graph on one DM.
export async function classifyLstm(text: string): Promise<ClassifyResult> {
  // Require loaded LSTM meta.
  const loaded = lstmMeta
  // The ORT session is the heavy singleton.
  const sessionHolder = getHeavySession()
  // Both must be present, LSTM family, same id.
  if (!loaded || !sessionHolder || sessionHolder.family !== 'lstm' || sessionHolder.id !== loaded.id) {
    // Fail rather than scoring with a stale vocab.
    throw new Error('LSTM checkpoint is not loaded')
  }
  // Time tokenize + LSTM path.
  const started = performance.now()
  // Unpadded token ids (batch=1).
  const ids = encodeLstmUnpadded(text, loaded.meta)
  // Scaled URL vector (zeros when there is no link).
  const url = scaleUrlFeatures(featuresToVector(extractMessageUrlFeatures(text)), loaded.scaler)
  // ORT wants int64 token ids.
  const tokenIds = new OrtTensor('int64', BigInt64Array.from(ids.map((id) => BigInt(id))), [1, ids.length])
  // ORT wants float32 URL features [1, 20].
  const urlFeatures = new OrtTensor('float32', url, [1, url.length])
  // Run the LSTM graph.
  const output = await sessionHolder.session.run({
    token_ids: tokenIds,
    url_features: urlFeatures,
  })
  // Logits are [1, 2].
  const logits = output.logits
  // Read the two class logits.
  const logit0 = Number(logits.data[0])
  // Scam logit is index 1.
  const logit1 = Number(logits.data[1])
  // Convert to P(scam).
  const pScam = softmaxScam(logit0, logit1)
  // Stop the timer.
  const inferenceMs = performance.now() - started
  // Apply the VAL-frozen threshold from the sidecar.
  const warned = pScam >= loaded.meta.threshold
  // Return the unified classify result.
  return { pScam, warned, checkpointId: loaded.id, inferenceMs }
}

// Fetch fixture_scores.json written at export time (Python/PyTorch reference).
export async function fetchFixtureScores(id: CheckpointId, filename: string): Promise<FixtureScore[]> {
  // Parse the fixture list used by the sequential banner check.
  return fetchJson<FixtureScore[]>(id, filename)
}

// ChatScreen eager default: TF-IDF Best (10k terms, C=1.0, threshold 0.20).
let chatDefaultReady: Promise<void> | null = null

// Load TF-IDF Best once (idempotent).
export function ensureChatDefaultClassifier(): Promise<void> {
  // Reuse the in-flight load so StrictMode double-mount does not fetch twice.
  if (chatDefaultReady === null) {
    // Load tfidf_best; failures surface to ChatScreen as a quiet no-banner state.
    chatDefaultReady = loadTfidfCheckpoint(CHATSCREEN_DEFAULT_ID).then(() => undefined)
  }
  // Return the memoized promise.
  return chatDefaultReady
}

// Classify verified plaintext with the selected heavy graph, else TF-IDF Best.
export async function classifyVerifiedPlaintext(
  text: string,
  heavy: ChatHeavyPreference,
): Promise<ClassifyResult | null> {
  // DistilBERT was requested: never silently score with TF-IDF while the graph is missing.
  if (heavy === 'distilbert') {
    // Score only when the Slice 5 DistilBERT default graph is actually loaded.
    if (distilbertVocab?.id === DISTILBERT_OPT_IN_ID) {
      try {
        // Run the transformer on this verified plaintext.
        return await classifyDistilbert(text)
      } catch {
        // WASM abort: skip the banner rather than falling through to TF-IDF.
        return null
      }
    }
    // Caller should retry after enableDistilbertOptIn; do not mix in TF-IDF banners.
    return null
  }
  // LSTM was requested: never silently score with TF-IDF while the graph is missing.
  if (heavy === 'lstm') {
    // Score only when the 8-epoch opt-in graph is actually loaded.
    if (lstmMeta?.id === LSTM_OPT_IN_ID) {
      try {
        // Run the word BiLSTM on this verified plaintext.
        return await classifyLstm(text)
      } catch {
        // WASM abort: skip the banner rather than falling through to TF-IDF.
        return null
      }
    }
    // Caller should retry after enableLstmOptIn; do not mix in TF-IDF banners.
    return null
  }
  // Eager TF-IDF Best path (A5).
  try {
    // Ensure TF-IDF Best is loaded.
    await ensureChatDefaultClassifier()
    // Score with the logistic head.
    return classifyTfidf(text)
  } catch {
    // Missing export or WASM abort: skip the banner rather than blocking chat.
    return null
  }
}

// Lazy-load Slice 5 DistilBERT default behind the ChatScreen opt-in toggle (A6).
export async function enableDistilbertOptIn(): Promise<void> {
  // Load the 256-token int8 graph (threshold 0.30); distilbert_best stays exported as a catalog row.
  await loadDistilbertCheckpoint(DISTILBERT_OPT_IN_ID)
}

// Unload DistilBERT so ChatScreen falls back to TF-IDF Best.
export async function disableDistilbertOptIn(): Promise<void> {
  // Only dispose DistilBERT; leave an LSTM session alone if that toggle is active.
  const heavy = getHeavySession()
  // Unload when the heavy singleton is the DistilBERT graph.
  if (heavy?.family === 'distilbert') {
    // Drop the transformer WASM session; TF-IDF stays loaded.
    await unloadHeavySession()
  }
  // Clear the WordPiece cache.
  distilbertVocab = null
}

// Lazy-load Word BiLSTM Best behind the ChatScreen opt-in toggle.
export async function enableLstmOptIn(): Promise<void> {
  // Load the 8-epoch sweep winner (not the published 4-epoch default).
  await loadLstmCheckpoint(LSTM_OPT_IN_ID)
}

// Unload the word BiLSTM so ChatScreen falls back to TF-IDF Best.
export async function disableLstmOptIn(): Promise<void> {
  // Only dispose LSTM; leave DistilBERT alone if that toggle is active.
  const heavy = getHeavySession()
  // Unload when the heavy singleton is the LSTM graph.
  if (heavy?.family === 'lstm') {
    // Drop the LSTM WASM session; TF-IDF stays loaded.
    await unloadHeavySession()
  }
  // Clear the LSTM tokenizer cache.
  lstmMeta = null
}

// Unload everything (sequential load-check between the six steps).
export async function resetAllClassifiers(): Promise<void> {
  // Dispose WASM sessions.
  await unloadAllSessions()
  // Drop sidecar caches so the next load cannot mix vocabs.
  tfidfSidecar = null
  // Drop DistilBERT vocab.
  distilbertVocab = null
  // Drop LSTM meta.
  lstmMeta = null
  // Allow ChatScreen to reload TF-IDF after a load-check.
  chatDefaultReady = null
}
