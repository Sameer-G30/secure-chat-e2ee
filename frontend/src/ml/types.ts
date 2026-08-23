// Shared types for ONNX Runtime Web checkpoints and classify results.

// The six sequential load-check ids; do not reorder in CHECKPOINT_LOAD_ORDER.
export type CheckpointId =
  | 'distilbert_best'
  | 'distilbert_default'
  | 'lstm_best'
  | 'lstm_default'
  | 'tfidf_best'
  | 'tfidf_default'

// Model family; ChatScreen may keep TF-IDF loaded beside one DistilBERT session.
export type CheckpointFamily = 'distilbert' | 'lstm' | 'tfidf'

// Manifest.json written by ml/src/secure_chat_ml/onnx_export.py.
export interface CheckpointManifest {
  // Catalog id matching CHECKPOINT_LOAD_ORDER.
  id: CheckpointId
  // Sequential check order (1..6).
  load_order: number
  // Which preprocessor + ONNX graph this folder contains.
  family: CheckpointFamily
  // Human-readable label for the load-check table.
  label: string
  // VAL-frozen P(scam) threshold; never retuned on chat_eval.
  threshold: number
  // DistilBERT truncation length; omitted for TF-IDF/LSTM.
  max_length?: number
  // ONNX filename relative to this folder.
  onnx_file: string
  // Serving graph size in bytes.
  onnx_bytes: number
  // int8 or fp32, recorded after export.
  quantize: 'int8' | 'fp32'
  // Sidecar filenames the TypeScript loaders fetch.
  sidecars: Record<string, string>
  // Per-file byte counts for the cost table.
  artifact_bytes: Record<string, number>
  // Offline TEST/chat-eval numbers cited beside browser measurements.
  offline: {
    reports_dir: string
    test_accuracy: number
    test_fn: number
    test_fp: number
    chat_accuracy: number | null
    chat_fn: number | null
    chat_fp: number | null
    note: string
  }
  // True only for the ChatScreen eager default (TF-IDF Best).
  wired_in_chatscreen_by_default: boolean
}

// One fixed fixture DM scored during export (not TEST accuracy).
export interface FixtureScore {
  // Stable fixture id (ham_no_url, scam_shortener, ...).
  id: string
  // Plaintext the browser classifies.
  text: string
  // Gold label for documentation only (0=legitimate, 1=scam).
  gold_label: number
  // Python/PyTorch P(scam) reference.
  p_scam: number
  // Whether the frozen threshold would show the banner.
  warned: boolean
  // Threshold used to compute `warned`.
  threshold: number
  // Optional DistilBERT input_ids for WordPiece debugging.
  input_ids?: number[]
}

// Result of classifying one verified plaintext string.
export interface ClassifyResult {
  // Scam-class probability from ONNX Runtime Web.
  pScam: number
  // True when pScam >= the frozen threshold.
  warned: boolean
  // Checkpoint that produced this score.
  checkpointId: CheckpointId
  // Wall-clock inference time for this message.
  inferenceMs: number
}

// One row of the six-way sequential load-check table.
export interface LoadCheckRow {
  // Catalog id.
  id: CheckpointId
  // Load order 1..6.
  loadOrder: number
  // Human-readable label.
  label: string
  // Whether InferenceSession.create succeeded.
  loadSuccess: boolean
  // Error text when load or inference failed.
  error: string | null
  // Session create latency in milliseconds.
  initMs: number | null
  // Mean inference milliseconds per fixture DM.
  inferenceMsPerMessage: number | null
  // Serving ONNX bytes from the manifest.
  onnxBytes: number | null
  // Approximate JS heap after load, when performance.memory exists.
  jsHeapBytes: number | null
  // Banner agreement vs Python fixture_scores (true/false per fixture).
  fixtureBannerMatch: boolean[] | null
  // Offline TEST/chat-eval citation copied from the manifest.
  offline: CheckpointManifest['offline'] | null
}

// Which heavy graph ChatScreen should score with; TF-IDF Best stays the eager fallback.
export type ChatHeavyPreference = 'tfidf' | 'distilbert' | 'lstm'

// ChatScreen eager default: TF-IDF Best (10k terms, C=1.0, threshold 0.20).
export const CHATSCREEN_DEFAULT_ID: CheckpointId = 'tfidf_best'

// DistilBERT opt-in loads the Slice 5 256-token graph, not the 512-token winner.
export const DISTILBERT_OPT_IN_ID: CheckpointId = 'distilbert_default'

// Word BiLSTM opt-in loads the 8-epoch sweep winner, not the published 4-epoch default.
export const LSTM_OPT_IN_ID: CheckpointId = 'lstm_best'

// Fixed sequential order; skip to the next row if this one OOMs.
export const CHECKPOINT_LOAD_ORDER: readonly CheckpointId[] = [
  'distilbert_best',
  'distilbert_default',
  'lstm_best',
  'lstm_default',
  'tfidf_best',
  'tfidf_default',
] as const
