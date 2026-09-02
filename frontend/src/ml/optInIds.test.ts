// Pin ChatScreen opt-in checkpoint ids so a catalog rename cannot silently switch DistilBERT graphs.

// Import Vitest helpers used by the rest of the frontend unit suite.
import { describe, expect, it } from 'vitest'

// Import the ChatScreen wiring constants that pick which ONNX folder to fetch.
import { CHATSCREEN_DEFAULT_ID, DISTILBERT_OPT_IN_ID, LSTM_OPT_IN_ID } from './types'

// Describe the eager vs lazy checkpoint contract.
describe('ChatScreen checkpoint ids', () => {
  // TF-IDF Best stays the tiny eager default; DistilBERT is still opt-in.
  it('keeps TF-IDF Best eager and DistilBERT default as the transformer opt-in', () => {
    // A5: the logistic head is always loaded on ChatScreen mount.
    expect(CHATSCREEN_DEFAULT_ID).toBe('tfidf_best')
    // Slice 5 256-token graph (threshold 0.30); the 512-token sweep winner stays a catalog export.
    expect(DISTILBERT_OPT_IN_ID).toBe('distilbert_default')
    // LSTM toggle still loads the 8-epoch winner, not the 4-epoch switch-back.
    expect(LSTM_OPT_IN_ID).toBe('lstm_best')
  })
})
