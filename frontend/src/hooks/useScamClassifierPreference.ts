// Per-account on-device scam model. The default is the MiniLM student.
// Settings can switch to DistilBERT best. Only one graph is loaded.
// The old secure-chat-classifier key is deleted so it cannot select TF-IDF or LSTM.

// Import React state/effect/ref hooks.
import { useCallback, useEffect, useRef, useState } from 'react'

// Import the browser classifier. TF-IDF and LSTM are not imported here.
import { classifyVerifiedPlaintext, ensureScamModel } from '../ml/scamClassifier'
import type { BrowserScamModelId, CheckpointId, ClassifyResult } from '../ml/types'
import { DISTILBERT_OPT_IN_ID, isBrowserScamModelId } from '../ml/types'

// Prefix of the old opt-in key. Chat load deletes it.
export const CLASSIFIER_STORAGE_PREFIX = 'secure-chat-classifier:'

// New per-account key. Values are distilbert_student or distilbert_best.
export const SCAM_MODEL_STORAGE_PREFIX = 'secure-chat-scam-model:'

// Build the per-user localStorage key this pass deletes.
export function classifierStorageKey(username: string): string {
  // Tokens never go in localStorage; the key only names the old model choice.
  return `${CLASSIFIER_STORAGE_PREFIX}${username}`
}

// Build the per-user key that stores the Settings choice.
export function scamModelStorageKey(username: string): string {
  // The value is a catalog id, never a message or a token.
  return `${SCAM_MODEL_STORAGE_PREFIX}${username}`
}

// Delete a stored tfidf/lstm/distilbert choice so it cannot trigger a fetch.
export function deleteClassifierPreference(username: string): void {
  try {
    // Remove the key. Missing keys throw nothing.
    window.localStorage.removeItem(classifierStorageKey(username))
  } catch {
    // Private mode must not break chat.
    return
  }
}

// Read the Settings choice. Missing or unknown values stay on the student.
export function readScamModelPreference(username: string): BrowserScamModelId {
  try {
    // Read only the new key. The old key is not a model choice anymore.
    const stored = window.localStorage.getItem(scamModelStorageKey(username))
    // Accept the two catalog ids and nothing else.
    if (isBrowserScamModelId(stored)) {
      // This account previously picked student or best.
      return stored
    }
  } catch {
    // Private mode falls through to the default.
    return DISTILBERT_OPT_IN_ID
  }
  // First visit, or a value this build does not understand.
  return DISTILBERT_OPT_IN_ID
}

// Persist the Settings choice under the new key.
export function writeScamModelPreference(username: string, modelId: BrowserScamModelId): void {
  try {
    // Store the catalog id only.
    window.localStorage.setItem(scamModelStorageKey(username), modelId)
  } catch {
    // The session can still switch for this page view.
    return
  }
}

// Describe what this hook hands back to ChatScreen.
export interface ScamClassifierPreferenceApi {
  // Load/failure status. Chat stays usable either way.
  classifierStatus: string | null
  // Increments when the resident graph changes so visible messages are re-scored.
  generation: number
  // True only after the chosen graph is resident. Banners stay off until then.
  ready: boolean
  // True after the chosen graph failed to load. Banners stay off.
  loadFailed: boolean
  // The Settings choice, which is the student until the account opts in.
  modelId: BrowserScamModelId
  // Score one verified string. Returns null until the chosen graph is ready.
  classify: (plaintext: string) => Promise<ClassifyResult | null>
  // Banner cache hits require this id. It follows the graph that actually loaded.
  scoringCheckpointId: CheckpointId
  // Save the choice, unload the current graph, and load the selected one.
  selectModel: (modelId: BrowserScamModelId) => void
}

// Load the account's scam model when the signed-in chat shell mounts.
export function useScamClassifierPreference(
  username: string | undefined,
): ScamClassifierPreferenceApi {
  // Student until storage says this account picked best.
  const [modelId, setModelId] = useState<BrowserScamModelId>(DISTILBERT_OPT_IN_ID)
  // True after ensureScamModel resolves for this choice.
  const [ready, setReady] = useState(false)
  // True when the last load rejected. Chat stays up and banners stay off.
  const [loadFailed, setLoadFailed] = useState(false)
  // Status line next to the transcript. Not a checkbox.
  const [classifierStatus, setClassifierStatus] = useState<string | null>(null)
  // Bump so useEncryptedConversation scores cache misses on the loaded graph.
  const [generation, setGeneration] = useState(0)
  // Cache key. Starts as the student so a reload can paint student banners before WASM.
  const [scoringCheckpointId, setScoringCheckpointId] = useState<CheckpointId>(DISTILBERT_OPT_IN_ID)
  // classify() reads this so a socket callback does not close over a stale ready flag.
  const readyRef = useRef(false)

  // Delete the old key and load the chosen graph when the account or the choice changes.
  useEffect(() => {
    // Logout render: leave banners off and do not touch storage for an empty handle.
    if (!username) {
      // classify() must not call into a graph after sign-out.
      readyRef.current = false
      // Drop the ready flag so a late render cannot score.
      setReady(false)
      // Stop.
      return
    }
    // An old "tfidf" or "lstm" value must not survive into this session.
    deleteClassifierPreference(username)
    // Storage wins over the initial student state on the first effect.
    const stored = readScamModelPreference(username)
    // Sync state and wait for the next effect rather than loading the wrong graph.
    if (stored !== modelId) {
      // The next run loads this id.
      setModelId(stored)
      // Do not also start a student download.
      return
    }
    // Ignore setState after this effect is cleaned up (StrictMode remount or a new choice).
    let cancelled = false
    // The graph is not ready yet. classify() returns null until the promise resolves.
    readyRef.current = false
    // Banners from a failed graph stay off while this attempt runs.
    setReady(false)
    // This attempt has not failed.
    setLoadFailed(false)
    // Cache lookups follow the requested id until a fallback reports a different resident id.
    setScoringCheckpointId(modelId)
    // Non-blocking status. Chat send/receive does not wait on this sentence.
    setClassifierStatus('Loading scam detection on this device…')
    // Invalidate scores from the previous graph before the new one is resident.
    setGeneration((current) => current + 1)
    // Unload the previous graph inside ensureScamModel, then load this one.
    void ensureScamModel(modelId)
      .then((residentId) => {
        // The first StrictMode mount is cancelled; the second owns the state.
        if (cancelled) {
          // Drop the result.
          return
        }
        // Scoring may now call the resident graph.
        readyRef.current = true
        // ChatScreen can classify.
        setReady(true)
        // A successful load clears the failure flag.
        setLoadFailed(false)
        // The student choice may have fallen back to distilbert_default.
        setScoringCheckpointId(residentId)
        // Re-score visible messages with the graph that is now resident.
        setGeneration((current) => current + 1)
        // Confirm on-device classification. Scores are not uploaded.
        setClassifierStatus('Scam detection is running on this device (not sent to the server).')
      })
      .catch(() => {
        // Drop a result for an unmounted tree or a choice that was replaced.
        if (cancelled) {
          // The newer effect owns the status line.
          return
        }
        // Failure leaves chat usable and banners off. No other model is loaded.
        readyRef.current = false
        // classify() keeps returning null.
        setReady(false)
        // The conversation hook clears visible banners.
        setLoadFailed(true)
        // Re-run scoring as a no-op so a previous graph's banner does not stick.
        setGeneration((current) => current + 1)
        // Say why the banners are off.
        setClassifierStatus(
          'Scam detection could not load on this device. Chat still works; warnings stay off.',
        )
      })
    // Cancel setState on the recycled mount. The shared load promise keeps running.
    return () => {
      // The next effect, if any, owns the following setState calls.
      cancelled = true
    }
  }, [username, modelId])

  // Save a new choice. The effect unloads the current graph and loads this one.
  const selectModel = useCallback(
    (next: BrowserScamModelId) => {
      // Signed-out settings cannot persist a choice.
      if (!username) {
        // Ignore the click.
        return
      }
      // Persist under the new key, not secure-chat-classifier.
      writeScamModelPreference(username, next)
      // Same choice: leave the resident graph alone.
      setModelId((current) => (current === next ? current : next))
    },
    [username],
  )

  // Score with the resident graph only. Null while it is still downloading or if it failed.
  const classify = useCallback(async (plaintext: string): Promise<ClassifyResult | null> => {
    // Empty plaintext has no scam signal.
    if (!plaintext) {
      // Nothing to score.
      return null
    }
    // Do not call the model, and do not fall back, until the graph is resident.
    if (!readyRef.current) {
      // Banners stay off.
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

  // Hand the shell the choice, the load state, and the scorer.
  return {
    classifierStatus,
    generation,
    ready,
    loadFailed,
    modelId,
    classify,
    scoringCheckpointId,
    selectModel,
  }
}
