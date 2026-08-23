// TypeScript TF-IDF that matches sklearn TfidfVectorizer (A5 browser path).
// The logistic head stays in ONNX; this module only builds the float vector.

// Import URL features so the FeatureUnion concat order stays tfidf then URL.
import { URL_FEATURE_NAMES, extractMessageUrlFeatures, featuresToVector, scaleUrlFeatures } from './urlFeatures'
import type { UrlScalerSidecar } from './urlFeatures'

// Sidecar written by ml/src/secure_chat_ml/onnx_export.py for each TF-IDF export.
export interface TfidfSidecar {
  // Index-ordered n-gram strings (column i is vocabulary_terms[i]).
  vocabulary_terms: string[]
  // idf_ aligned with vocabulary_terms (smooth_idf log form).
  idf: number[]
  // Inclusive lower n-gram order (published default is 1).
  ngram_min: number
  // Inclusive upper n-gram order (published default is 2).
  ngram_max: number
  // When true, replace TF with 1 + log(TF) before IDF.
  sublinear_tf: boolean
  // When true, multiply by the exported idf_ row.
  use_idf: boolean
  // Lowercase before tokenization (sklearn default True).
  lowercase: boolean
  // Accent folding: "unicode" matches sklearn.strip_accents_unicode.
  strip_accents: string | null
  // Documented sklearn token_pattern; implemented via Unicode word runs.
  token_pattern: string
  // Vector norm applied to the TF-IDF block only (not the URL concat).
  norm: string
  // Analyzer name; only "word" is implemented.
  analyzer: string
  // Vocabulary width (must equal vocabulary_terms.length).
  n_vocab: number
  // URL feature width (must equal 20).
  n_url: number
  // Logistic coef_ flattened to length n_vocab + n_url.
  coef: number[]
  // Logistic intercept_ scalar for the binary scam class.
  intercept: number
  // Frozen C recorded for the README, not used at inference.
  C: number
  // VAL-frozen P(scam) cut.
  threshold: number
  // Must stay false: no HTTP reputation lookups.
  live_url_reputation: boolean
}

// Match sklearn's (?u)\b\w\w+\b using Unicode letters/digits/underscore, length >= 2.
const WORD_TOKEN_RE = /[\p{L}\p{N}_]{2,}/gu

// Drop Unicode combining marks after NFKD, matching sklearn strip_accents_unicode.
function stripAccentsUnicode(text: string): string {
  // NFKD splits base characters from combining accents.
  const normalized = text.normalize('NFKD')
  // Remove combining marks so café → cafe, matching sklearn.
  return normalized.replace(/\p{M}/gu, '')
}

// Apply sklearn TfidfVectorizer preprocessor: lowercase then optional accent strip.
export function preprocessTfidfText(text: string, sidecar: TfidfSidecar): string {
  // Start from the raw DM (already decrypted plaintext on this device).
  let doc = text
  // Lowercase first, matching sklearn's preprocessor order.
  if (sidecar.lowercase) {
    // Case folding is locale-independent, same as Python str.lower for DMs.
    doc = doc.toLowerCase()
  }
  // Unicode accent stripping is the published baseline default.
  if (sidecar.strip_accents === 'unicode') {
    // Fold accents after lowercasing.
    doc = stripAccentsUnicode(doc)
  }
  // Return the preprocessed string the tokenizer consumes.
  return doc
}

// Tokenize like sklearn's default word analyzer (tokens of 2+ word characters).
export function tokenizeTfidf(text: string, sidecar: TfidfSidecar): string[] {
  // Preprocess before token_pattern so accents/case cannot split n-grams.
  const prepared = preprocessTfidfText(text, sidecar)
  // Collect every 2+ character Unicode word run.
  const tokens = prepared.match(WORD_TOKEN_RE)
  // Empty DMs produce no tokens; the TF-IDF vector stays zeros then L2-noops.
  return tokens ?? []
}

// Build unigram/bigram strings using sklearn's space-joined n-gram format.
export function buildNgrams(tokens: string[], ngramMin: number, ngramMax: number): string[] {
  // Accumulate n-grams in document order (counts, not unique).
  const ngrams: string[] = []
  // Walk every requested n in the inclusive range.
  for (let n = ngramMin; n <= ngramMax; n += 1) {
    // Skip n larger than the token list (sklearn emits nothing for those).
    if (n > tokens.length) {
      // Continue so a shorter n still emits.
      continue
    }
    // Slide a window of length n across the token list.
    for (let start = 0; start <= tokens.length - n; start += 1) {
      // sklearn joins n-gram tokens with a single space.
      ngrams.push(tokens.slice(start, start + n).join(' '))
    }
  }
  // Return the multiset as a list; callers count frequencies.
  return ngrams
}

// L2-normalize a dense TF-IDF row in place; a zero vector stays zero.
function l2Normalize(values: Float32Array): void {
  // Accumulate squared magnitude.
  let sumSquares = 0
  // Walk every TF-IDF coordinate (URL features are concatenated after this).
  for (let index = 0; index < values.length; index += 1) {
    // Add the squared coordinate.
    const value = values[index] ?? 0
    // Track the squared L2 norm.
    sumSquares += value * value
  }
  // sklearn returns the zero vector unchanged when the norm is 0.
  if (sumSquares <= 0) {
    // Nothing to scale.
    return
  }
  // Divide each coordinate by the L2 norm.
  const norm = Math.sqrt(sumSquares)
  // Scale in place.
  for (let index = 0; index < values.length; index += 1) {
    // Each TF-IDF weight is divided by ||tfidf||_2.
    values[index] = (values[index] ?? 0) / norm
  }
}

// Build the FeatureUnion dense row: L2 TF-IDF concatenated with scaled URL features.
export function vectorizeTfidfUnion(
  text: string,
  sidecar: TfidfSidecar,
  scaler: UrlScalerSidecar,
): Float32Array {
  // Tokenize with the sklearn-compatible word analyzer.
  const tokens = tokenizeTfidf(text, sidecar)
  // Expand to unigrams+bigrams (or whatever ngram_min/max the sidecar recorded).
  const ngrams = buildNgrams(tokens, sidecar.ngram_min, sidecar.ngram_max)
  // Count term frequencies in this document only.
  const tf = new Map<string, number>()
  // Increment the count for every n-gram occurrence.
  for (const gram of ngrams) {
    // Missing keys start at 0.
    tf.set(gram, (tf.get(gram) ?? 0) + 1)
  }
  // Allocate the TF-IDF block (zeros for n-grams not in this DM).
  const tfidf = new Float32Array(sidecar.n_vocab)
  // Fill only the columns that appear in this document.
  for (const [gram, count] of tf) {
    // Look up the sklearn column index; OOV n-grams are dropped.
    const column = sidecar.vocabulary_terms.indexOf(gram)
    // Skip n-grams that missed the TRAIN vocabulary cap.
    if (column < 0) {
      // OOV: same as sklearn transform.
      continue
    }
    // Raw term frequency.
    let weight = count
    // sublinear_tf replaces TF with 1 + log(TF) when TF > 0.
    if (sidecar.sublinear_tf && count > 0) {
      // Natural log, matching sklearn.
      weight = 1 + Math.log(count)
    }
    // Multiply by the TRAIN idf_ value when IDF is enabled.
    if (sidecar.use_idf) {
      // idf_ is aligned with vocabulary_terms.
      weight *= sidecar.idf[column] ?? 0
    }
    // Store the unnormalized TF-IDF weight.
    tfidf[column] = weight
  }
  // L2-normalize the TF-IDF block only; URL features are not part of this norm.
  if (sidecar.norm === 'l2') {
    // sklearn TfidfVectorizer.norm='l2' default.
    l2Normalize(tfidf)
  }
  // Extract the 20-d raw URL vector (zeros when the DM has no link).
  const rawUrl = featuresToVector(extractMessageUrlFeatures(text))
  // Apply the TRAIN-fitted StandardScaler.
  const scaledUrl = scaleUrlFeatures(rawUrl, scaler)
  // Concatenate in FeatureUnion order: TF-IDF columns, then URL columns.
  const combined = new Float32Array(sidecar.n_vocab + URL_FEATURE_NAMES.length)
  // Copy the L2-normalized TF-IDF block into the left side.
  combined.set(tfidf, 0)
  // Copy the scaled URL block into the right side.
  combined.set(scaledUrl, sidecar.n_vocab)
  // Return the dense float row the logistic ONNX head consumes.
  return combined
}

// Map term → column once so vectorizeTfidfUnion is not O(vocab) per n-gram.
export function buildTfidfVocabularyIndex(sidecar: TfidfSidecar): Map<string, number> {
  // Allocate the reverse lookup used by the hot path.
  const index = new Map<string, number>()
  // Fill token → column for every TRAIN n-gram.
  for (let column = 0; column < sidecar.vocabulary_terms.length; column += 1) {
    // vocabulary_terms[i] is the n-gram for logistic column i.
    const term = sidecar.vocabulary_terms[column]
    // Skip empty holes that should not exist in a well-formed export.
    if (term) {
      // Record the sklearn column index.
      index.set(term, column)
    }
  }
  // Return the map the optimized vectorizer uses.
  return index
}

// Faster vectorizer that uses a prebuilt term → column map (50k vocab).
export function vectorizeTfidfUnionIndexed(
  text: string,
  sidecar: TfidfSidecar,
  scaler: UrlScalerSidecar,
  vocabIndex: Map<string, number>,
): Float32Array {
  // Tokenize with the sklearn-compatible word analyzer.
  const tokens = tokenizeTfidf(text, sidecar)
  // Expand to the sidecar's n-gram range.
  const ngrams = buildNgrams(tokens, sidecar.ngram_min, sidecar.ngram_max)
  // Count term frequencies in this document only.
  const tf = new Map<string, number>()
  // Increment the count for every n-gram occurrence.
  for (const gram of ngrams) {
    // Missing keys start at 0.
    tf.set(gram, (tf.get(gram) ?? 0) + 1)
  }
  // Allocate the TF-IDF block.
  const tfidf = new Float32Array(sidecar.n_vocab)
  // Fill only the columns that appear in this document.
  for (const [gram, count] of tf) {
    // O(1) TRAIN vocabulary lookup.
    const column = vocabIndex.get(gram)
    // Skip n-grams that missed the TRAIN vocabulary cap.
    if (column === undefined) {
      // OOV: same as sklearn transform.
      continue
    }
    // Raw term frequency.
    let weight = count
    // sublinear_tf replaces TF with 1 + log(TF) when TF > 0.
    if (sidecar.sublinear_tf && count > 0) {
      // Natural log, matching sklearn.
      weight = 1 + Math.log(count)
    }
    // Multiply by the TRAIN idf_ value when IDF is enabled.
    if (sidecar.use_idf) {
      // idf_ is aligned with vocabulary_terms.
      weight *= sidecar.idf[column] ?? 0
    }
    // Store the unnormalized TF-IDF weight.
    tfidf[column] = weight
  }
  // L2-normalize the TF-IDF block only.
  if (sidecar.norm === 'l2') {
    // sklearn TfidfVectorizer.norm='l2' default.
    l2Normalize(tfidf)
  }
  // Extract the 20-d raw URL vector.
  const rawUrl = featuresToVector(extractMessageUrlFeatures(text))
  // Apply the TRAIN-fitted StandardScaler.
  const scaledUrl = scaleUrlFeatures(rawUrl, scaler)
  // Concatenate in FeatureUnion order.
  const combined = new Float32Array(sidecar.n_vocab + URL_FEATURE_NAMES.length)
  // Copy TF-IDF into the left side.
  combined.set(tfidf, 0)
  // Copy scaled URL features into the right side.
  combined.set(scaledUrl, sidecar.n_vocab)
  // Return the dense float row the logistic ONNX head consumes.
  return combined
}
