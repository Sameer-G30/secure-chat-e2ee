// Exercise the word-BiLSTM tokenizer ported from ml/src/secure_chat_ml/lstm.py.

import { encodeLstmUnpadded, tokenizeLstmText } from './lstmTokenizer'
import type { LstmMetaSidecar } from './lstmTokenizer'
import { describe, expect, it } from 'vitest'

describe('lstmTokenizer', () => {
  it('splits alphanumeric runs and punctuation the way Python tokenize_text does', () => {
    // URLs explode into short pieces; that is why URL features are concatenated.
    const tokens = tokenizeLstmText('Hello, https://bit.ly/x')
    // Lowercase, comma as its own token, scheme punctuation exploded.
    expect(tokens[0]).toBe('hello')
    expect(tokens).toContain(',')
    expect(tokens).toContain(':')
    expect(tokens).toContain('/')
  })

  it('maps OOV tokens to UNK and uses PAD for empty text', () => {
    // Tiny vocab matching PAD/UNK conventions.
    const meta: LstmMetaSidecar = {
      architecture: 'word_bilstm_url_concat',
      token_to_id: { '<pad>': 0, '<unk>': 1, 'hello': 2 },
      pad_index: 0,
      unk_index: 1,
      max_tokens: 8,
      url_feature_names: [],
      url_dim: 20,
      scaler_mean: [],
      scaler_scale: [],
      threshold: 0.3,
      embed_dim: 128,
      hidden_size: 128,
      num_layers: 1,
      live_url_reputation: false,
      wired_in_chatscreen: false,
    }
    // Empty DMs still need length 1 for the ONNX LSTM.
    expect(encodeLstmUnpadded('', meta)).toEqual([0])
    // hello is in vocab; world is not.
    expect(encodeLstmUnpadded('hello world', meta)).toEqual([2, 1])
  })
})
