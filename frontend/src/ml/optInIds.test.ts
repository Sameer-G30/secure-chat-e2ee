// The chat bundle loads one DistilBERT-family graph: the student, or distilbert_best from Settings.

// Import the classifier and load-check sources as text so this test can see imports.
import classifierSource from './scamClassifier.ts?raw'
import loadCheckSource from './loadCheck.ts?raw'
// Import Vitest helpers.
import { describe, expect, it } from 'vitest'

// Import the checkpoint ids the chat shell can score with.
import { CHATSCREEN_DEFAULT_ID, DISTILBERT_BEST_ID, DISTILBERT_OPT_IN_ID } from './types'

describe('browser scam model', () => {
  it('defaults to the student, can load distilbert_best, and does not import TF-IDF or LSTM', () => {
    // The banner cache starts on the student.
    expect(CHATSCREEN_DEFAULT_ID).toBe('distilbert_student')
    // The default Settings choice is the student.
    expect(DISTILBERT_OPT_IN_ID).toBe('distilbert_student')
    // Settings may switch to the 512-token graph. It is not a second resident session.
    expect(DISTILBERT_BEST_ID).toBe('distilbert_best')
    // The loader names that opt-in so the switch can fetch it.
    expect(classifierSource).toContain('distilbert_best')
    // No TF-IDF vectorizer import, so Vite will not fetch that graph from this module.
    expect(classifierSource).not.toContain("from './tfidf'")
    // No LSTM tokenizer import.
    expect(classifierSource).not.toContain("from './lstmTokenizer'")
    // The chat loader does not name the TF-IDF or LSTM folders.
    expect(classifierSource).not.toContain('tfidf_best')
    expect(classifierSource).not.toContain('lstm_best')
    // The load-check page is the other entry that used to walk all six graphs.
    expect(loadCheckSource).not.toContain('loadTfidfCheckpoint')
    expect(loadCheckSource).not.toContain('loadLstmCheckpoint')
    expect(loadCheckSource).not.toContain('tfidf_best')
    expect(loadCheckSource).not.toContain('lstm_')
  })
})