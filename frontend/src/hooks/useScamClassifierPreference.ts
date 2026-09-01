// Extracted from ChatScreen.tsx during the pre-deployment refactor (Phase 3a). Owns the
// DistilBERT / Word BiLSTM Best lazy opt-in toggles (mutually exclusive; TF-IDF Best is
// always the eager fallback), their load status, and the classify() entry point every
// verified-plaintext message is run through. Behavior is unchanged from before the split.

// Import React's state/effect/ref hooks.
import { useEffect, useRef, useState } from 'react'

// Import the on-device scam classifier (TF-IDF Best eager; DistilBERT / BiLSTM opt-in).
import {
  classifyVerifiedPlaintext,
  disableDistilbertOptIn,
  disableLstmOptIn,
  enableDistilbertOptIn,
  enableLstmOptIn,
  ensureChatDefaultClassifier,
} from '../ml/scamClassifier'
import type { ChatHeavyPreference, ClassifyResult } from '../ml/types'

// Describe what this hook hands back to ChatScreen.
export interface ScamClassifierPreferenceApi {
  // Whether the operator opted into lazy-loaded DistilBERT (A6).
  useDistilbert: boolean
  // Whether the operator opted into lazy-loaded Word BiLSTM Best.
  useLstm: boolean
  // Load/failure status shown next to the toggles without blocking chat.
  classifierStatus: string | null
  // Flip the DistilBERT checkbox, loading or unloading the heavy model as needed.
  toggleDistilbert: (enabled: boolean) => void
  // Flip the Word BiLSTM checkbox, loading or unloading the heavy model as needed.
  toggleLstm: (enabled: boolean) => void
  // Classify one verified plaintext string with whichever model is currently selected.
  classify: (plaintext: string) => Promise<ClassifyResult | null>
}

// Manage the eager/opt-in scam-classifier selection for one ChatScreen session.
export function useScamClassifierPreference(): ScamClassifierPreferenceApi {
  const [useDistilbert, setUseDistilbert] = useState(false)
  const [useLstm, setUseLstm] = useState(false)
  const [classifierStatus, setClassifierStatus] = useState<string | null>(null)

  // Keep the selected classifier in a ref so incoming envelopes (decrypted inside a
  // WebSocket callback, not a render) always classify with the current model.
  const heavyPreferenceRef = useRef<ChatHeavyPreference>('tfidf')

  // Mirror the model toggles into the ref used by socket callbacks.
  useEffect(() => {
    heavyPreferenceRef.current = useDistilbert ? 'distilbert' : useLstm ? 'lstm' : 'tfidf'
  }, [useDistilbert, useLstm])

  // Eager-load TF-IDF Best once ChatScreen mounts (A5).
  useEffect(() => {
    void ensureChatDefaultClassifier().catch(() => {
      // Missing ONNX export must not block the encrypted conversation.
    })
  }, [])

  function toggleDistilbert(enabled: boolean) {
    if (enabled) {
      setUseLstm(false)
      setUseDistilbert(true)
      setClassifierStatus('Loading DistilBERT on this device…')
      void enableDistilbertOptIn()
        .then(() => {
          setClassifierStatus('DistilBERT is classifying on this device (not sent to the server).')
        })
        .catch((caught: unknown) => {
          setUseDistilbert(false)
          setClassifierStatus(
            caught instanceof Error
              ? `DistilBERT failed to load: ${caught.message}`
              : 'DistilBERT failed to load; staying on TF-IDF Best.',
          )
        })
    } else {
      setUseDistilbert(false)
      void disableDistilbertOptIn()
      setClassifierStatus('Using the on-device TF-IDF Best classifier.')
    }
  }

  function toggleLstm(enabled: boolean) {
    if (enabled) {
      setUseDistilbert(false)
      setUseLstm(true)
      setClassifierStatus('Loading Word BiLSTM Best on this device…')
      void enableLstmOptIn()
        .then(() => {
          setClassifierStatus(
            'Word BiLSTM Best is classifying on this device (not sent to the server).',
          )
        })
        .catch((caught: unknown) => {
          setUseLstm(false)
          setClassifierStatus(
            caught instanceof Error
              ? `Word BiLSTM Best failed to load: ${caught.message}`
              : 'Word BiLSTM Best failed to load; staying on TF-IDF Best.',
          )
        })
    } else {
      setUseLstm(false)
      void disableLstmOptIn()
      setClassifierStatus('Using the on-device TF-IDF Best classifier.')
    }
  }

  // Classify verified plaintext with whichever model is currently selected. Skip empty
  // strings; they have no scam signal and would still hit ORT for nothing.
  async function classify(plaintext: string): Promise<ClassifyResult | null> {
    if (!plaintext) {
      return null
    }
    return classifyVerifiedPlaintext(plaintext, heavyPreferenceRef.current)
  }

  return { useDistilbert, useLstm, classifierStatus, toggleDistilbert, toggleLstm, classify }
}
