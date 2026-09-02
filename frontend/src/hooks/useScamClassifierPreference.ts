// Owns DistilBERT default / Word BiLSTM Best lazy opt-in (mutually exclusive; TF-IDF Best
// is the eager fallback), load status, and classify() for verified plaintext.
//
// The checkbox may be checked while the heavy graph is still downloading. Scoring must
// not silently fall back to TF-IDF in that window (TF-IDF warns far more ham, and
// banners used to latch on). heavyPreferenceRef and classify() only request DistilBERT
// or LSTM after that graph is actually loaded. ChatScreen re-scores cache misses
// whenever `generation` increments; hits reuse last session's banners without WASM.

// Import React state/effect/ref hooks.
import { useCallback, useEffect, useRef, useState } from 'react'

// Import the on-device scam classifier (TF-IDF Best eager; DistilBERT / BiLSTM opt-in).
import {
  classifyVerifiedPlaintext,
  disableDistilbertOptIn,
  disableLstmOptIn,
  enableDistilbertOptIn,
  enableLstmOptIn,
  ensureChatDefaultClassifier,
} from '../ml/scamClassifier'
import type { ChatHeavyPreference, CheckpointId, ClassifyResult } from '../ml/types'
import { CHATSCREEN_DEFAULT_ID, DISTILBERT_OPT_IN_ID, LSTM_OPT_IN_ID } from '../ml/types'

// Prefix localStorage keys so the opt-in is per signed-in username, not shared.
export const CLASSIFIER_STORAGE_PREFIX = 'secure-chat-classifier:'

// Persist which heavy graph this username last asked for (tfidf means neither checkbox).
export type StoredClassifierPreference = ChatHeavyPreference

// Build the per-user localStorage key for the scam-classifier opt-in.
export function classifierStorageKey(username: string): string {
  // Tokens never go in localStorage; only the model name is stored.
  return `${CLASSIFIER_STORAGE_PREFIX}${username}`
}

// Read the saved opt-in for this username, defaulting to eager TF-IDF Best.
export function loadClassifierPreference(username: string): StoredClassifierPreference {
  try {
    // Read the per-user key.
    const stored = window.localStorage.getItem(classifierStorageKey(username))
    // Only accept the three documented values.
    if (stored === 'distilbert' || stored === 'lstm' || stored === 'tfidf') {
      // Restore the last explicit choice after reload.
      return stored
    }
  } catch {
    // Private mode or blocked storage must not break chat.
  }
  // Unset or unreadable storage means the tiny eager default.
  return 'tfidf'
}

// Persist the opt-in for this username only.
export function saveClassifierPreference(
  username: string,
  preference: StoredClassifierPreference,
): void {
  try {
    // Store the model name, never a session token.
    window.localStorage.setItem(classifierStorageKey(username), preference)
  } catch {
    // A storage failure leaves the in-memory choice in place for this session.
  }
}

// Describe what this hook hands back to ChatScreen.
export interface ScamClassifierPreferenceApi {
  // Whether the operator asked for DistilBERT (checkbox); may still be loading.
  useDistilbert: boolean
  // Whether the operator asked for Word BiLSTM Best (checkbox); may still be loading.
  useLstm: boolean
  // Load/failure status shown next to the toggles without blocking chat.
  classifierStatus: string | null
  // Increments whenever the *ready* scoring model changes so history can be re-scored.
  generation: number
  // Flip the DistilBERT checkbox, loading or unloading the heavy model as needed.
  toggleDistilbert: (enabled: boolean) => void
  // Flip the Word BiLSTM checkbox, loading or unloading the heavy model as needed.
  toggleLstm: (enabled: boolean) => void
  // Classify one verified plaintext string with the ready model only (never a silent TF-IDF mix).
  classify: (plaintext: string) => Promise<ClassifyResult | null>
  // Catalog id whose cached banners may be painted before that graph is in WASM.
  scoringCheckpointId: CheckpointId
}

// Manage the eager/opt-in scam-classifier selection for one ChatScreen session.
export function useScamClassifierPreference(
  username: string | undefined,
): ScamClassifierPreferenceApi {
  // Checkbox intent: DistilBERT requested (true while the 64 MiB graph downloads).
  const [useDistilbert, setUseDistilbert] = useState(false)
  // Checkbox intent: LSTM requested.
  const [useLstm, setUseLstm] = useState(false)
  // True only after enableDistilbertOptIn resolved for the current request.
  const [distilbertReady, setDistilbertReady] = useState(false)
  // True only after enableLstmOptIn resolved for the current request.
  const [lstmReady, setLstmReady] = useState(false)
  // Load/failure copy shown next to the toggles.
  const [classifierStatus, setClassifierStatus] = useState<string | null>(null)
  // Bump so useEncryptedConversation re-scores visible verified bubbles.
  const [generation, setGeneration] = useState(0)

  // Ignore a stale DistilBERT/LSTM load that finished after the user toggled again.
  const loadTokenRef = useRef(0)
  // Latest checkbox + ready flags for classify() (socket callbacks close over a stale function).
  const useDistilbertRef = useRef(false)
  // Latest LSTM checkbox intent for classify().
  const useLstmRef = useRef(false)
  // Latest DistilBERT-ready flag for classify().
  const distilbertReadyRef = useRef(false)
  // Latest LSTM-ready flag for classify().
  const lstmReadyRef = useRef(false)
  // Ready scoring model only (never 'distilbert' while the graph is still downloading).
  const heavyPreferenceRef = useRef<ChatHeavyPreference>('tfidf')

  // Mirror React state into refs before any classify() call in this render's effects.
  useDistilbertRef.current = useDistilbert
  // Keep the LSTM checkbox ref in lockstep.
  useLstmRef.current = useLstm
  // Keep the DistilBERT-ready ref in lockstep.
  distilbertReadyRef.current = distilbertReady
  // Keep the LSTM-ready ref in lockstep.
  lstmReadyRef.current = lstmReady
  // DistilBERT/LSTM are requested only when both the checkbox and the load have succeeded.
  heavyPreferenceRef.current =
    useDistilbert && distilbertReady ? 'distilbert' : useLstm && lstmReady ? 'lstm' : 'tfidf'

  // Eager-load TF-IDF Best once ChatScreen mounts (A5).
  useEffect(() => {
    void ensureChatDefaultClassifier().catch(() => {
      // Missing ONNX export must not block the encrypted conversation.
    })
  }, [])

  // Restore this username's last opt-in after login or reload.
  useEffect(() => {
    // No session yet (logout render): leave checkboxes off.
    if (!username) {
      // Stop so we do not read localStorage for an empty handle.
      return
    }
    // Read the persisted choice (defaults to tfidf).
    const stored = loadClassifierPreference(username)
    // Invalidate any in-flight load from a previous username.
    loadTokenRef.current += 1
    // Capture the token this restore's async load must match.
    const token = loadTokenRef.current
    if (stored === 'distilbert') {
      // Show the checkbox as checked while the graph downloads.
      setUseDistilbert(true)
      // LSTM cannot be selected at the same time.
      setUseLstm(false)
      // Do not score with DistilBERT until enableDistilbertOptIn resolves.
      setDistilbertReady(false)
      // LSTM is not the restored choice.
      setLstmReady(false)
      // Tell the operator the 37 MiB download is in progress.
      setClassifierStatus('Loading DistilBERT on this device…')
      // Drop in-flight TF-IDF scores; classify() returns null until the graph is ready.
      setGeneration((current) => current + 1)
      void enableDistilbertOptIn()
        .then(() => {
          // Drop a restore that finished after the user toggled or logged out.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Scoring may now request DistilBERT default.
          setDistilbertReady(true)
          // Re-score history that was held back (or TF-IDF-scored) during the download.
          setGeneration((current) => current + 1)
          // Confirm on-device classification.
          setClassifierStatus(
            'DistilBERT is classifying on this device (not sent to the server).',
          )
        })
        .catch((caught: unknown) => {
          // Drop a restore that finished after the user toggled or logged out.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Uncheck so the UI matches TF-IDF fallback.
          setUseDistilbert(false)
          // DistilBERT did not load.
          setDistilbertReady(false)
          // Persist the fallback so the next reload does not retry a broken graph forever.
          saveClassifierPreference(username, 'tfidf')
          // Re-score with TF-IDF Best.
          setGeneration((current) => current + 1)
          // Show the failure next to the toggle.
          setClassifierStatus(
            caught instanceof Error
              ? `DistilBERT failed to load: ${caught.message}`
              : 'DistilBERT failed to load; staying on TF-IDF Best.',
          )
        })
      // Restored DistilBERT; skip the LSTM/tfidf branches.
      return
    }
    if (stored === 'lstm') {
      // Show the LSTM checkbox as checked while the graph downloads.
      setUseLstm(true)
      // DistilBERT cannot be selected at the same time.
      setUseDistilbert(false)
      // Do not score with LSTM until enableLstmOptIn resolves.
      setLstmReady(false)
      // DistilBERT is not the restored choice.
      setDistilbertReady(false)
      // Tell the operator the LSTM download is in progress.
      setClassifierStatus('Loading Word BiLSTM Best on this device…')
      // Drop in-flight TF-IDF scores; classify() returns null until the graph is ready.
      setGeneration((current) => current + 1)
      void enableLstmOptIn()
        .then(() => {
          // Drop a restore that finished after the user toggled or logged out.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Scoring may now request Word BiLSTM Best.
          setLstmReady(true)
          // Re-score history held back during the download.
          setGeneration((current) => current + 1)
          // Confirm on-device classification.
          setClassifierStatus(
            'Word BiLSTM Best is classifying on this device (not sent to the server).',
          )
        })
        .catch((caught: unknown) => {
          // Drop a restore that finished after the user toggled or logged out.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Uncheck so the UI matches TF-IDF fallback.
          setUseLstm(false)
          // LSTM did not load.
          setLstmReady(false)
          // Persist the fallback.
          saveClassifierPreference(username, 'tfidf')
          // Re-score with TF-IDF Best.
          setGeneration((current) => current + 1)
          // Show the failure next to the toggle.
          setClassifierStatus(
            caught instanceof Error
              ? `Word BiLSTM Best failed to load: ${caught.message}`
              : 'Word BiLSTM Best failed to load; staying on TF-IDF Best.',
          )
        })
      // Restored LSTM; skip the tfidf reset.
      return
    }
    // Stored tfidf (or missing): both checkboxes off, TF-IDF already eager-loaded.
    setUseDistilbert(false)
    // LSTM checkbox off.
    setUseLstm(false)
    // DistilBERT is not loaded for scoring.
    setDistilbertReady(false)
    // LSTM is not loaded for scoring.
    setLstmReady(false)
    // Unload a previous account's heavy graph; skip on first mount (nothing is loaded).
    if (distilbertReadyRef.current || useDistilbertRef.current) {
      // Dispose DistilBERT WASM so the new username cannot inherit it.
      void disableDistilbertOptIn()
    }
    // Unload a previous account's LSTM graph the same way.
    if (lstmReadyRef.current || useLstmRef.current) {
      // Dispose LSTM WASM so the new username cannot inherit it.
      void disableLstmOptIn()
    }
    // Clear opt-in status copy from a previous account on this tab.
    setClassifierStatus(null)
  }, [username])

  // Bump generation so the open transcript is re-scored with the newly ready model.
  function bumpGeneration(): void {
    // Functional update so rapid toggle/load completions cannot clobber each other.
    setGeneration((current) => current + 1)
  }

  function toggleDistilbert(enabled: boolean) {
    // Invalidate any in-flight DistilBERT or LSTM load.
    loadTokenRef.current += 1
    // Capture the token this click's async load must match.
    const token = loadTokenRef.current
    if (enabled) {
      // Uncheck LSTM; only one heavy graph is resident.
      setUseLstm(false)
      // LSTM scoring must stop immediately.
      setLstmReady(false)
      // Show DistilBERT as checked while it downloads.
      setUseDistilbert(true)
      // Do not score with DistilBERT until the graph is in WASM.
      setDistilbertReady(false)
      // Persist so a reload restores this checkbox and waits for the graph.
      if (username) {
        // Store distilbert, not a token.
        saveClassifierPreference(username, 'distilbert')
      }
      // Tell the operator the download started.
      setClassifierStatus('Loading DistilBERT on this device…')
      // Drop in-flight TF-IDF results so they cannot latch banners after this click.
      bumpGeneration()
      void enableDistilbertOptIn()
        .then(() => {
          // Drop a load that finished after uncheck / LSTM / logout.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Scoring may now request DistilBERT default.
          setDistilbertReady(true)
          // Re-score bubbles that were skipped or TF-IDF-scored before this load.
          bumpGeneration()
          // Confirm on-device classification.
          setClassifierStatus(
            'DistilBERT is classifying on this device (not sent to the server).',
          )
        })
        .catch((caught: unknown) => {
          // Drop a load that finished after uncheck / LSTM / logout.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Uncheck so the UI matches TF-IDF fallback.
          setUseDistilbert(false)
          // DistilBERT did not load.
          setDistilbertReady(false)
          // Persist the fallback.
          if (username) {
            // Store tfidf so reload does not retry forever.
            saveClassifierPreference(username, 'tfidf')
          }
          // Re-score with TF-IDF Best.
          bumpGeneration()
          // Show the failure next to the toggle.
          setClassifierStatus(
            caught instanceof Error
              ? `DistilBERT failed to load: ${caught.message}`
              : 'DistilBERT failed to load; staying on TF-IDF Best.',
          )
        })
    } else {
      // Uncheck DistilBERT.
      setUseDistilbert(false)
      // Stop DistilBERT scoring immediately (classify() will use TF-IDF).
      setDistilbertReady(false)
      // Unload the heavy WASM session; TF-IDF stays loaded.
      void disableDistilbertOptIn()
      // Persist eager TF-IDF.
      if (username) {
        // Store tfidf.
        saveClassifierPreference(username, 'tfidf')
      }
      // Re-score visible bubbles with TF-IDF Best (and allow banners to clear).
      bumpGeneration()
      // Confirm the fallback.
      setClassifierStatus('Using the on-device TF-IDF Best classifier.')
    }
  }

  function toggleLstm(enabled: boolean) {
    // Invalidate any in-flight DistilBERT or LSTM load.
    loadTokenRef.current += 1
    // Capture the token this click's async load must match.
    const token = loadTokenRef.current
    if (enabled) {
      // Uncheck DistilBERT; only one heavy graph is resident.
      setUseDistilbert(false)
      // DistilBERT scoring must stop immediately.
      setDistilbertReady(false)
      // Show LSTM as checked while it downloads.
      setUseLstm(true)
      // Do not score with LSTM until the graph is in WASM.
      setLstmReady(false)
      // Persist so a reload restores this checkbox.
      if (username) {
        // Store lstm, not a token.
        saveClassifierPreference(username, 'lstm')
      }
      // Tell the operator the download started.
      setClassifierStatus('Loading Word BiLSTM Best on this device…')
      // Drop in-flight TF-IDF results so they cannot latch banners after this click.
      bumpGeneration()
      void enableLstmOptIn()
        .then(() => {
          // Drop a load that finished after uncheck / DistilBERT / logout.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Scoring may now request Word BiLSTM Best.
          setLstmReady(true)
          // Re-score bubbles held back during the download.
          bumpGeneration()
          // Confirm on-device classification.
          setClassifierStatus(
            'Word BiLSTM Best is classifying on this device (not sent to the server).',
          )
        })
        .catch((caught: unknown) => {
          // Drop a load that finished after uncheck / DistilBERT / logout.
          if (token !== loadTokenRef.current) {
            // A newer request owns the checkboxes now.
            return
          }
          // Uncheck so the UI matches TF-IDF fallback.
          setUseLstm(false)
          // LSTM did not load.
          setLstmReady(false)
          // Persist the fallback.
          if (username) {
            // Store tfidf.
            saveClassifierPreference(username, 'tfidf')
          }
          // Re-score with TF-IDF Best.
          bumpGeneration()
          // Show the failure next to the toggle.
          setClassifierStatus(
            caught instanceof Error
              ? `Word BiLSTM Best failed to load: ${caught.message}`
              : 'Word BiLSTM Best failed to load; staying on TF-IDF Best.',
          )
        })
    } else {
      // Uncheck LSTM.
      setUseLstm(false)
      // Stop LSTM scoring immediately.
      setLstmReady(false)
      // Unload the heavy WASM session; TF-IDF stays loaded.
      void disableLstmOptIn()
      // Persist eager TF-IDF.
      if (username) {
        // Store tfidf.
        saveClassifierPreference(username, 'tfidf')
      }
      // Re-score visible bubbles with TF-IDF Best.
      bumpGeneration()
      // Confirm the fallback.
      setClassifierStatus('Using the on-device TF-IDF Best classifier.')
    }
  }

  // Classify verified plaintext with the ready model only. Skip empty strings.
  const classify = useCallback(async (plaintext: string): Promise<ClassifyResult | null> => {
    // Empty plaintext has no scam signal and would still hit ORT.
    if (!plaintext) {
      // Leave the banner unchanged.
      return null
    }
    // DistilBERT checkbox on but graph not in WASM: do not silently use TF-IDF.
    if (useDistilbertRef.current && !distilbertReadyRef.current) {
      // History is re-scored when generation bumps after load.
      return null
    }
    // LSTM checkbox on but graph not in WASM: same hold-back.
    if (useLstmRef.current && !lstmReadyRef.current) {
      // History is re-scored when generation bumps after load.
      return null
    }
    try {
      // Score with the ready model (tfidf, distilbert, or lstm).
      return await classifyVerifiedPlaintext(plaintext, heavyPreferenceRef.current)
    } catch {
      // WASM abort must not reject into the conversation hook.
      return null
    }
  }, [])

  // Cache lookups use the requested graph even while DistilBERT/LSTM is still loading.
  const scoringCheckpointId: CheckpointId = useDistilbert
    ? DISTILBERT_OPT_IN_ID
    : useLstm
      ? LSTM_OPT_IN_ID
      : CHATSCREEN_DEFAULT_ID

  return {
    useDistilbert,
    useLstm,
    classifierStatus,
    generation,
    toggleDistilbert,
    toggleLstm,
    classify,
    scoringCheckpointId,
  }
}
