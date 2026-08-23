// Exercise DistilBERT WordPiece encoding against a tiny local vocab.

import { basicTokenize, buildWordpieceIndex, encodeWordpiece } from './wordpiece'
import type { WordpieceVocab } from './wordpiece'
import { describe, expect, it } from 'vitest'

describe('wordpiece', () => {
  it('lowercases and splits punctuation in BasicTokenizer', () => {
    // HuggingFace DistilBERT-base is uncased and splits commas.
    const tokens = basicTokenize('Hello, World', true)
    // Lowercase plus comma as its own token.
    expect(tokens).toEqual(['hello', ',', 'world'])
  })

  it('wraps WordPiece ids with [CLS] and [SEP] and pads to max_length', () => {
    // Tiny vocab with DistilBERT special-token ids.
    const vocab: WordpieceVocab = {
      tokens: ['[PAD]', 'hello', '[UNK]', '[CLS]', '[SEP]'],
      unk_token: '[UNK]',
      cls_token: '[CLS]',
      sep_token: '[SEP]',
      pad_token: '[PAD]',
      unk_id: 2,
      cls_id: 3,
      sep_id: 4,
      pad_id: 0,
      do_lower_case: true,
      max_length: 8,
      max_input_chars_per_word: 100,
    }
    // Build the reverse map the encoder uses.
    const index = buildWordpieceIndex(vocab)
    // Encode a single in-vocab token.
    const encoded = encodeWordpiece('hello', vocab, index)
    // [CLS] hello [SEP] then pad.
    expect(encoded.inputIds.slice(0, 3)).toEqual([3, 1, 4])
    // Padded to max_length.
    expect(encoded.inputIds).toHaveLength(8)
    // Attention mask is 1 for the three real tokens.
    expect(encoded.attentionMask.slice(0, 3)).toEqual([1, 1, 1])
    expect(encoded.attentionMask.slice(3)).toEqual([0, 0, 0, 0, 0])
  })
})
