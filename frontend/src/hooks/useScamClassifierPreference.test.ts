// Storage key for the Settings scam-model choice. The classifier module is mocked.

// Import Vitest helpers.
import { describe, expect, it, vi } from 'vitest'

// Avoid loading ONNX just to check localStorage.
vi.mock('../ml/scamClassifier', () => ({
  classifyVerifiedPlaintext: vi.fn(),
  ensureScamModel: vi.fn(async (id: string) => id),
}))

// Import the key helpers under test.
import {
  CLASSIFIER_STORAGE_PREFIX,
  SCAM_MODEL_STORAGE_PREFIX,
  deleteClassifierPreference,
  readScamModelPreference,
  scamModelStorageKey,
  writeScamModelPreference,
} from './useScamClassifierPreference'

describe('scam model preference storage', () => {
  // The new key is not the old secure-chat-classifier key.
  it('persists distilbert_best under the new key and drops the old one', () => {
    // The prefixes differ so an old tfidf value cannot select a graph.
    expect(SCAM_MODEL_STORAGE_PREFIX).not.toBe(CLASSIFIER_STORAGE_PREFIX)
    // An old choice left by a previous build.
    window.localStorage.setItem('secure-chat-classifier:alice', 'lstm')
    // Chat load deletes it.
    deleteClassifierPreference('alice')
    // Gone.
    expect(window.localStorage.getItem('secure-chat-classifier:alice')).toBeNull()
    // Nothing stored yet means the student.
    expect(readScamModelPreference('alice')).toBe('distilbert_student')
    // Settings picks best.
    writeScamModelPreference('alice', 'distilbert_best')
    // The new key holds the catalog id.
    expect(window.localStorage.getItem(scamModelStorageKey('alice'))).toBe('distilbert_best')
    // The reader returns that choice.
    expect(readScamModelPreference('alice')).toBe('distilbert_best')
    // A legacy value in the new key is ignored.
    window.localStorage.setItem(scamModelStorageKey('alice'), 'tfidf')
    // Fall back to the student.
    expect(readScamModelPreference('alice')).toBe('distilbert_student')
  })
})
