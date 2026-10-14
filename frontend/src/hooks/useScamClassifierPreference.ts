// Loads DistilBERT default when the signed-in chat shell mounts.
// There is no model checkbox and no second graph. TF-IDF and the word BiLSTM
// are not fetched. An old secure-chat-classifier:<username> value is deleted
// so it cannot select them.

// Import React state/effect/ref hooks.
import { useCallback, useEffect, useRef, useState } from 'react'

// Import the only browser classifier.
import { classifyVerifiedPlaintext, ensureDistilbertDefault } from '../ml/scamClassifier'
import type { CheckpointId, ClassifyResult } from '../ml/types'
import { DISTILBERT_OPT_IN_ID } from '../ml/types'

// Prefix of the old opt-in key. Chat load deletes it.
export const CLASSIFIER_STORAGE_PREFIX = 'secure-chat-classifier:'

// Build the per-user localStorage key this pass deletes.
export function classifierStorageKey(username: string): string {
  // Tokens never go in localStorage; the key only names the old model choice.
  return `${CLASSIFIER_STORAGE_PREFIX}${username}`
}

// Delete a stored tfidf/lstm/distilbert choice so it cannot trigger a fetch.
export function deleteClassifierPreference(username: string): void {
  try {
    // Remove the key. Missing keys throw nothing.
    window.localStorage.removeItem(classifierStorageKey(username))
  } catch {
    // Private mode must not break chat.
  }
}

// Describe what this hook hands back to ChatScreen.
export interface ScamClassifierPreferenceApi {
  // Load/failure status. Chat stays usable either way.
  classifierStatus: string | null
  // Increments when DistilBERT default becomes ready so cache misses can be scored.
  generation: number
  // True only after distilbert_default is resident. Banners stay off until then.
  ready: boolean
  // Score one verified string. Returns null until DistilBERT default is ready.
  classify: (plaintext: string) => Promise<ClassifyResult | null>
  // Banner cache hits require this id. It does not change when the graph is still downloading.
  scoringCheckpointId: CheckpointId
}

// Load DistilBERT default for one signed-in chat shell.
export function useScamClassifierPreference(
  username: string | undefined,
): ScamClassifierPreferenceApi {
  // True after ensureDistilbertDefault resolves for this mount.
  const [ready, setReady] = useState(false)
  // Status line next to the transcript. Not a checkbox.
  const [classifierStatus, setClassifierStatus] = useState<string | null>(null)
  // Bump so useEncryptedConversation scores cache misses on the loaded page.
  const [generation, setGeneration] = useState(0)
  // classify() reads this so a socket callback does not close over a stale ready flag.
  const readyRef = useRef(false)

  // Delete the old key and load DistilBERT default when a username is present.
  useEffect(() => {
    // Logout render: leave banners off and do not touch storage for an empty handle.
    if (!username) {
      return
    }
    // An old "tfidf" or "lstm" value must not survive into this session.
    deleteClassifierPreference(username)
    // Ignore setState after this effect is cleaned up (StrictMode remount).
    let cancelled = false
    // The graph is not ready yet. classify() returns null until the promise resolves.
    readyRef.current = false
    setReady(false)
    // Non-blocking status. Chat send/receive does not wait on this sentence.
    setClassifierStatus('Loading scam detection on this device…')
    // Share one in-flight load with a StrictMode remount.
    void ensureDistilbertDefault()
      .then(() => {
        // The first StrictMode mount is cancelled; the second owns the state.
        if (cancelled) {
          return
        }
        // Scoring may now call DistilBERT default.
        readyRef.current = true
        setReady(true)
        // Re-score cache misses that arrived while the graph was downloading.
        setGeneration((current) => current + 1)
        // Confirm on-device classification. Scores are not uploaded.
        setClassifierStatus('Scam detection is running on this device (not sent to the server).')
      })
      .catch(() => {
        // Drop a result for an unmounted tree.
        if (cancelled) {
          return
        }
        // Failure leaves chat usable and banners off. No other model is loaded.
        readyRef.current = false
        setReady(false)
        setClassifierStatus(
          'Scam detection could not load on this device. Chat still works; warnings stay off.',
        )
      })
    // Cancel setState on the recycled mount. The shared load promise keeps running.
    return () => {
      cancelled = true
    }
  }, [username])

  // Score with DistilBERT default only. Null while it is still downloading or if it failed.
  const classify = useCallback(async (plaintext: string): Promise<ClassifyResult | null> => {
    // Empty plaintext has no scam signal.
    if (!plaintext) {
      return null
    }
    // Do not call the model, and do not fall back, until the graph is resident.
    if (!readyRef.current) {
      return null
    }
    try {
      // classifyVerifiedPlaintext itself no-ops if the session is missing.
      return await classifyVerifiedPlaintext(plaintext)
    } catch {
      // A thrown WASM error must not reject into the conversation hook.
      return null
    }
  }, [])

  return {
    classifierStatus,
    generation,
    ready,
    classify,
    // Cache hits require distilbert_default even while the download is in progress.
    scoringCheckpointId: DISTILBERT_OPT_IN_ID,
  }
}
