// The chat bundle loads DistilBERT default only.

// Import the classifier and load-check sources as text so this test can see imports.
import classifierSource from './scamClassifier.ts?raw'
import loadCheckSource from './loadCheck.ts?raw'
// Import Vitest helpers.
import { describe, expect, it } from 'vitest'

// Import the checkpoint id the chat shell scores with.
import { CHATSCREEN_DEFAULT_ID, DISTILBERT_OPT_IN_ID } from './types'

describe('browser scam model', () => {
  it('loads DistilBERT default and does not import TF-IDF or LSTM loaders', () => {
    // The banner cache and the session both use this id.
    expect(CHATSCREEN_DEFAULT_ID).toBe('distilbert_default')
    // The 512-token sweep winner is not the browser graph.
    expect(DISTILBERT_OPT_IN_ID).toBe('distilbert_default')
    // No TF-IDF vectorizer import, so Vite will not fetch that graph from this module.
    expect(classifierSource).not.toContain("from './tfidf'")
    // No LSTM tokenizer import.
    expect(classifierSource).not.toContain("from './lstmTokenizer'")
    // The chat loader does not name the other checkpoint folders.
    expect(classifierSource).not.toContain('tfidf_best')
    expect(classifierSource).not.toContain('lstm_best')
    expect(classifierSource).not.toContain('distilbert_best')
    // The load-check page is the other entry that used to walk all six graphs.
    expect(loadCheckSource).not.toContain('loadTfidfCheckpoint')
    expect(loadCheckSource).not.toContain('loadLstmCheckpoint')
    expect(loadCheckSource).not.toContain('tfidf_best')
    expect(loadCheckSource).not.toContain('lstm_')
  })
})