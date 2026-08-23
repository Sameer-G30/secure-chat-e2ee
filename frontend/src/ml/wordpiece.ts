// DistilBERT WordPiece tokenizer matching HuggingFace DistilBertTokenizer.
// The ONNX graph consumes input_ids + attention_mask; tokenization stays in JS (A6).

// Sidecar written by write_wordpiece_vocab in ml/src/secure_chat_ml/onnx_export.py.
export interface WordpieceVocab {
  // Id-ordered WordPiece strings, including ##subwords and special tokens.
  tokens: string[]
  // Unknown-token string, usually [UNK].
  unk_token: string
  // Classification token prepended to every sequence.
  cls_token: string
  // Separator token appended to every sequence.
  sep_token: string
  // Padding token used when we pad to max_length.
  pad_token: string
  // Integer ids for the special tokens (DistilBERT: 100/101/102/0).
  unk_id: number
  cls_id: number
  sep_id: number
  pad_id: number
  // DistilBERT-base is uncased.
  do_lower_case: boolean
  // Truncation length for this checkpoint (256 or 512).
  max_length: number
  // HuggingFace WordPiece limit; longer tokens become UNK.
  max_input_chars_per_word: number
}

// Encoded tensors DistilBERT ONNX expects (int64 in ORT, number[] here).
export interface WordpieceEncoding {
  // Token ids including [CLS] / [SEP] / [PAD].
  inputIds: number[]
  // 1 for real tokens, 0 for pad.
  attentionMask: number[]
}

// Control characters are stripped; tab/newline/carriage-return stay as whitespace.
function isControlChar(char: string): boolean {
  // Preserve whitespace that BasicTokenizer later collapses to a space.
  if (char === '\t' || char === '\n' || char === '\r') {
    // Not treated as a control character to delete.
    return false
  }
  // Unicode category C* matches HuggingFace _is_control.
  return /^\p{C}$/u.test(char)
}

// ASCII punctuation plus any Unicode category P* character.
function isPunctuationChar(char: string): boolean {
  // HuggingFace treats ASCII non-alnum ranges as punctuation even if not P*.
  const code = char.charCodeAt(0)
  // Match BertBasicTokenizer._is_punctuation ASCII ranges.
  if (
    (code >= 33 && code <= 47) ||
    (code >= 58 && code <= 64) ||
    (code >= 91 && code <= 96) ||
    (code >= 123 && code <= 126)
  ) {
    // ASCII punctuation always splits.
    return true
  }
  // Unicode punctuation categories (P*).
  return /^\p{P}$/u.test(char)
}

// Whitespace that BasicTokenizer uses as a token boundary.
function isWhitespaceChar(char: string): boolean {
  // Explicit ASCII whitespace plus Unicode separator spaces.
  return char === ' ' || char === '\t' || char === '\n' || char === '\r' || /^\p{Zs}$/u.test(char)
}

// CJK code points that HuggingFace isolates with spaces before WordPiece.
function isChineseChar(code: number): boolean {
  // CJK Unified Ideographs and common extension blocks used by BertTokenizer.
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x20000 && code <= 0x2a6df) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0x2f800 && code <= 0x2fa1f)
  )
}

// Drop invalid/control characters and collapse whitespace, matching _clean_text.
function cleanText(text: string): string {
  // Accumulate surviving characters.
  let output = ''
  // Walk every UTF-16 code unit; DistilBERT DMs are BMP-heavy.
  for (const char of text) {
    // Delete NUL and replacement character, matching HuggingFace.
    const code = char.codePointAt(0) ?? 0
    // Skip NUL, U+FFFD, and control characters.
    if (code === 0 || code === 0xfffd || isControlChar(char)) {
      // Drop this character.
      continue
    }
    // Map every whitespace character to a single space.
    if (isWhitespaceChar(char)) {
      // Collapse happens later via split.
      output += ' '
    } else {
      // Keep letters, punctuation, and digits.
      output += char
    }
  }
  // Return the cleaned string.
  return output
}

// Insert spaces around CJK characters so WordPiece sees them as isolated tokens.
function tokenizeChineseChars(text: string): string {
  // Accumulate with optional spaces around CJK code points.
  let output = ''
  // Walk each character.
  for (const char of text) {
    // Read the code point for the CJK range check.
    const code = char.codePointAt(0) ?? 0
    // Isolate CJK ideographs.
    if (isChineseChar(code)) {
      // Surround with spaces, matching HuggingFace _tokenize_chinese_chars.
      output += ` ${char} `
    } else {
      // Keep non-CJK characters unchanged.
      output += char
    }
  }
  // Return the spaced string.
  return output
}

// Strip combining marks after NFKD, matching BertTokenizer._run_strip_accents.
function stripAccents(text: string): string {
  // NFKD splits base characters from combining accents.
  const normalized = text.normalize('NFKD')
  // Drop combining marks.
  return normalized.replace(/\p{M}/gu, '')
}

// Split a whitespace token on punctuation, matching _run_split_on_punc.
function splitOnPunctuation(token: string): string[] {
  // Accumulate characters of the current alphanumeric run.
  const output: string[] = []
  // Buffer for the current non-punctuation run.
  let current = ''
  // Walk every character.
  for (const char of token) {
    // Punctuation becomes its own token.
    if (isPunctuationChar(char)) {
      // Flush the buffered run first.
      if (current) {
        // Push the alphanumeric/other run.
        output.push(current)
        // Reset the buffer.
        current = ''
      }
      // Punctuation is a singleton token.
      output.push(char)
    } else {
      // Continue the current run.
      current += char
    }
  }
  // Flush a trailing run.
  if (current) {
    // Push the final run.
    output.push(current)
  }
  // Return the punctuation-split pieces.
  return output
}

// Bert BasicTokenizer: clean, isolate CJK, lowercase, strip accents, split punct.
export function basicTokenize(text: string, doLowerCase: boolean): string[] {
  // Clean control characters and normalize whitespace.
  const cleaned = cleanText(text)
  // Isolate CJK characters.
  const spaced = tokenizeChineseChars(cleaned)
  // Split on whitespace.
  const origTokens = spaced.trim() ? spaced.trim().split(/\s+/) : []
  // Accumulate punctuation-split tokens.
  const splitTokens: string[] = []
  // Process each whitespace token.
  for (let token of origTokens) {
    // Lowercase when the DistilBERT checkpoint is uncased.
    if (doLowerCase) {
      // Lowercase then strip accents, matching HuggingFace order.
      token = stripAccents(token.toLowerCase())
    }
    // Split punctuation away from words.
    splitTokens.push(...splitOnPunctuation(token))
  }
  // Return the BasicTokenizer output WordPiece consumes.
  return splitTokens
}

// Greedy longest-match WordPiece using the exported vocab list.
export function wordpieceTokenize(
  token: string,
  tokenToId: Map<string, number>,
  unkToken: string,
  maxInputCharsPerWord: number,
): string[] {
  // Tokens longer than the HuggingFace cap become a single UNK.
  if (token.length > maxInputCharsPerWord) {
    // Return UNK rather than an exploding subword list.
    return [unkToken]
  }
  // Accumulate subword pieces for this BasicTokenizer token.
  const subTokens: string[] = []
  // Greedy scan from start to end.
  let start = 0
  // True when any span cannot be matched.
  let isBad = false
  // Walk until the whole token is consumed.
  while (start < token.length) {
    // End index of the current candidate span (exclusive).
    let end = token.length
    // Best vocab match for this start position.
    let curSubstr: string | null = null
    // Shrink the span until it hits the vocabulary.
    while (start < end) {
      // Continuation pieces use the ## prefix.
      let substr = token.slice(start, end)
      // Prefix ## when this is not the first subword.
      if (start > 0) {
        // HuggingFace WordPiece continuation marker.
        substr = `##${substr}`
      }
      // Accept the longest span present in the vocab.
      if (tokenToId.has(substr)) {
        // Record the match and stop shrinking.
        curSubstr = substr
        // Leave the inner loop.
        break
      }
      // Shrink from the right.
      end -= 1
    }
    // No span from this start is in vocab → whole token is UNK.
    if (curSubstr === null) {
      // Mark the token as un-segmentable.
      isBad = true
      // Stop the outer scan.
      break
    }
    // Keep the matched piece.
    subTokens.push(curSubstr)
    // Advance past the matched span (without the ## prefix length).
    start = end
  }
  // Un-segmentable tokens become UNK, matching HuggingFace.
  if (isBad) {
    // Single UNK for the whole BasicTokenizer token.
    return [unkToken]
  }
  // Return the WordPiece subwords.
  return subTokens
}

// Build a reverse vocab map once per DistilBERT load.
export function buildWordpieceIndex(vocab: WordpieceVocab): Map<string, number> {
  // Allocate token → id.
  const index = new Map<string, number>()
  // Fill from the id-ordered array.
  for (let id = 0; id < vocab.tokens.length; id += 1) {
    // Skip empty holes.
    const token = vocab.tokens[id]
    // Only index non-empty strings.
    if (token) {
      // Record the HuggingFace id.
      index.set(token, id)
    }
  }
  // Return the map encodeWordpiece uses on every message.
  return index
}

// Encode one DM to DistilBERT input_ids and attention_mask (truncate, pad).
export function encodeWordpiece(
  text: string,
  vocab: WordpieceVocab,
  tokenToId: Map<string, number>,
): WordpieceEncoding {
  // Run BasicTokenizer with the sidecar's lowercasing flag.
  const basic = basicTokenize(text, vocab.do_lower_case)
  // Accumulate WordPiece strings including special tokens.
  const pieces: string[] = [vocab.cls_token]
  // WordPiece each BasicTokenizer token.
  for (const token of basic) {
    // Greedy longest-match subwords.
    const sub = wordpieceTokenize(
      token,
      tokenToId,
      vocab.unk_token,
      vocab.max_input_chars_per_word,
    )
    // Append subwords before the final [SEP].
    pieces.push(...sub)
  }
  // Append the separator token.
  pieces.push(vocab.sep_token)
  // Truncate to max_length, keeping [CLS] and forcing [SEP] at the end.
  const maxLength = vocab.max_length
  // HuggingFace truncates then ensures special tokens still fit.
  if (pieces.length > maxLength) {
    // Keep max_length - 1 tokens then force [SEP] as the last id.
    pieces.length = maxLength - 1
    // Restore the separator that truncation may have dropped.
    pieces.push(vocab.sep_token)
  }
  // Map pieces to ids; unknown strings become unk_id.
  const inputIds = pieces.map((piece) => tokenToId.get(piece) ?? vocab.unk_id)
  // Attention mask is 1 for every real token before padding.
  const attentionMask = inputIds.map(() => 1)
  // Pad to max_length so the DistilBERT graph sees a rectangular tensor.
  while (inputIds.length < maxLength) {
    // Pad id is 0 for DistilBERT.
    inputIds.push(vocab.pad_id)
    // Pad positions are masked out.
    attentionMask.push(0)
  }
  // Return tensors the ORT DistilBERT session consumes.
  return { inputIds, attentionMask }
}
