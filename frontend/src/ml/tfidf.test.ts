// Exercise sklearn-compatible TF-IDF tokenization used by the A5 browser path.

import { buildNgrams, preprocessTfidfText, tokenizeTfidf } from './tfidf'
import type { TfidfSidecar } from './tfidf'
import { describe, expect, it } from 'vitest'

// Minimal sidecar so tokenizer tests do not need a 50k vocab dump.
const sidecar: TfidfSidecar = {
  vocabulary_terms: ['hello', 'hello world', 'world'],
  idf: [1, 1, 1],
  ngram_min: 1,
  ngram_max: 2,
  sublinear_tf: true,
  use_idf: true,
  lowercase: true,
  strip_accents: 'unicode',
  token_pattern: '(?u)\\b\\w\\w+\\b',
  norm: 'l2',
  analyzer: 'word',
  n_vocab: 3,
  n_url: 20,
  coef: [0, 0, 0],
  intercept: 0,
  C: 0.25,
  threshold: 0.3,
  live_url_reputation: false,
}

describe('tfidf', () => {
  it('lowercases and strips unicode accents before tokenizing', () => {
    // café should become cafe so it matches sklearn strip_accents='unicode'.
    const prepared = preprocessTfidfText('Café', sidecar)
    // Accent folding happens after lowercase.
    expect(prepared).toBe('cafe')
  })

  it('drops one-character tokens like sklearn token_pattern \\w\\w+', () => {
    // "a hello b" should keep only hello.
    const tokens = tokenizeTfidf('a hello b', sidecar)
    // Single-character tokens are not sklearn word tokens.
    expect(tokens).toEqual(['hello'])
  })

  it('builds space-joined bigrams like sklearn', () => {
    // Two tokens produce one bigram plus two unigrams.
    const ngrams = buildNgrams(['hello', 'world'], 1, 2)
    // sklearn order is all n=1 then all n=2 as we implemented... wait.
    // Our implementation walks n from min to max, so unigrams first.
    expect(ngrams).toEqual(['hello', 'world', 'hello world'])
  })
})
