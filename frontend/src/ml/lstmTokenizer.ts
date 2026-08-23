// Word tokenizer matching ml/src/secure_chat_ml/lstm.py tokenize_text + encode_texts.
// ONNX consumes unpadded token ids (batch=1, dynamic length) plus scaled URL features.

// Sidecar written beside each word-BiLSTM ONNX graph.
export interface LstmMetaSidecar {
  // Architecture tag so a future char-LSTM cannot be loaded by accident.
  architecture: string
  // TRAIN token → embedding row index, including <pad> and <unk>.
  token_to_id: Record<string, number>
  // PAD must stay 0 for embedding padding_idx.
  pad_index: number
  // UNK must stay 1 so OOV hosts share one vector.
  unk_index: number
  // Truncation length in word tokens (published default 128).
  max_tokens: number
  // Frozen URL feature names; must match url_features.py.
  url_feature_names: string[]
  // Width of the URL concat (0 if an ablation dropped URL features).
  url_dim: number
  // TRAIN StandardScaler mean_.
  scaler_mean: number[]
  // TRAIN StandardScaler scale_.
  scaler_scale: number[]
  // VAL-frozen P(scam) threshold.
  threshold: number
  // Embedding width recorded for the README, not used at inference.
  embed_dim: number
  // Per-direction LSTM hidden size.
  hidden_size: number
  // Stacked LSTM layer count (published default 1).
  num_layers: number
  // Must stay false: no HTTP reputation lookups.
  live_url_reputation: boolean
  // Slice 6 does not wire LSTM into ChatScreen.
  wired_in_chatscreen: boolean
}

// Split a message into lowercase alphanumeric runs and single punctuation tokens.
export function tokenizeLstmText(text: string): string[] {
  // Missing or empty text produces no tokens; encodeLstmUnpadded inserts PAD.
  if (!text) {
    // Return an empty list so callers can test emptiness uniformly.
    return []
  }
  // Lowercase first so TRAIN/VAL/TEST share one vocabulary case.
  const lowered = text.toLowerCase()
  // Find alphanumeric runs or single non-whitespace, non-alnum characters.
  const tokens = lowered.match(/[a-z0-9]+|[^a-z0-9\s]/g)
  // Empty after stripping whitespace still yields no tokens.
  return tokens ?? []
}

// Encode one DM as unpadded token ids (min length 1) for the ONNX LSTM graph.
export function encodeLstmUnpadded(text: string, meta: LstmMetaSidecar): number[] {
  // Split with the documented whitespace/punctuation tokenizer.
  let tokens = tokenizeLstmText(text)
  // Truncate on the right when the DM overflows max_tokens.
  if (tokens.length > meta.max_tokens) {
    // Keep the leading tokens; DistilBERT also truncates the tail.
    tokens = tokens.slice(0, meta.max_tokens)
  }
  // Map each token through the TRAIN vocab, defaulting to UNK.
  const ids = tokens.map((token) => meta.token_to_id[token] ?? meta.unk_index)
  // Empty DMs still need length >= 1 so the LSTM sees one timestep.
  if (ids.length === 0) {
    // A lone PAD token matches encode_texts' empty-message convention.
    return [meta.pad_index]
  }
  // Return the unpadded id row the ONNX graph consumes.
  return ids
}
