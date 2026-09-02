// ChatScreen scoring policy: a short conversation window, a skip for trivial
// URL-free lines, and a slightly higher banner threshold than the frozen VAL cut.
// Ciphertext never enters this module; callers pass already-verified plaintext.

// Reuse the on-device URL extractor so a short "t.co/..." DM is never skipped.
import { extractUrls } from './urlFeatures'
// Reuse ClassifyResult so the ChatScreen overlay can flip `warned` without a second type.
import type { ClassifyResult } from './types'

// Number of recent verified turns (including the current DM) fed to the model.
// Six sits in the requested 5–8 range: enough multi-turn scam setup, few enough
// tokens that DistilBERT-256 / LSTM-128 still keep the current message.
export const CONTEXT_WINDOW_MESSAGES = 6

// Character cap after which oldest turns are dropped so truncation hits context,
// not the DM we are actually scoring. DistilBERT keeps a 256-token head; ~4
// characters per token leaves headroom for WordPiece-exploded URLs.
export const CONTEXT_MAX_CHARS = 900

// Skip classify only for tiny chit-chat with no link ("ok", "lol", "yes", "thanks").
// 8 characters is conservative: "click here" (10) and "hey it's me" (11) still score.
export const TRIVIAL_MAX_CHARS = 8

// ChatScreen banner cut. Frozen VAL thresholds stay on the ONNX sidecars
// (tfidf_best 0.20, distilbert_default 0.30, lstm_best 0.20) for load-check
// agreement with Python. ChatScreen then requires P(scam) to also clear 0.35
// so ordinary DMs warn less often while multi-turn context recovers split scams.
export const CHATSCREEN_MIN_SCAM_THRESHOLD = 0.35

// One verified bubble the window can concatenate. Matches ChatMessage's
// scoring-relevant fields without importing the conversation hook.
export interface ConversationTurn {
  // Stable id so the current DM can be excluded from the prior-turn list.
  id: string
  // Speaker role used only as a local prefix; never sent to the server.
  direction: 'sent' | 'received'
  // Verified plaintext, or null when AEAD failed (those turns are dropped).
  plaintext: string | null
  // True when decrypt+verify failed; those rows must never reach the model.
  verificationFailed: boolean
}

// True when the on-device extractor found at least one web URL in this DM.
export function messageHasUrl(text: string): boolean {
  // extractUrls never fetches; it only scans the string.
  return extractUrls(text).length > 0
}

// True when this single DM is too small to score on its own and has no URL.
export function isTrivialShortNoUrl(text: string): boolean {
  // Work from trimmed text so padding spaces cannot dodge the length cap.
  const trimmed = text.trim()
  // Empty strings have no scam signal; classify() already no-ops on them.
  if (!trimmed) {
    // Treat empty as skip so we never pay WASM for a blank bubble.
    return true
  }
  // A URL, even on an otherwise tiny line, is enough reason to run the model.
  if (messageHasUrl(trimmed)) {
    // Shorteners and paste-only links must not be skipped.
    return false
  }
  // Digits keep OTPs, amounts, and phone numbers in the scoring path.
  if (/\d/.test(trimmed)) {
    // "otp 12" / "send 500" are short but not trivial chit-chat.
    return false
  }
  // Only URL-free, digit-free lines at or under the cap are skipped.
  return trimmed.length <= TRIVIAL_MAX_CHARS
}

// Format one turn as "Me: ..." or "Them: ..." so the concatenated window
// preserves who said what without leaving the device.
export function formatConversationTurn(turn: ConversationTurn, plaintext: string): string {
  // Sent bubbles are this tab; received bubbles are the peer.
  const speaker = turn.direction === 'sent' ? 'Me' : 'Them'
  // Speaker label plus the verified body; newline joining happens at the window.
  return `${speaker}: ${plaintext}`
}

// Verified turns strictly before `currentId` (oldest first). The current DM is
// supplied separately so an in-flight send/edit is not duplicated or stale.
export function collectPriorTurns(
  transcript: readonly ConversationTurn[],
  currentId: string,
): ConversationTurn[] {
  // Accumulate verified plaintext rows until we hit the current id.
  const prior: ConversationTurn[] = []
  // Walk oldest → newest so slice(-(window-1)) later keeps the most recent prior.
  for (const turn of transcript) {
    // Stop once we reach the DM being scored so its old plaintext cannot leak in.
    if (turn.id === currentId) {
      // Later rows are replies after this DM; they must not leak into its score.
      break
    }
    // Verification failures have no trustworthy plaintext.
    if (turn.verificationFailed || turn.plaintext === null) {
      // Drop the row rather than concatenating a fake body.
      continue
    }
    // Keep this verified prior turn for the window.
    prior.push(turn)
  }
  // Return the full prior list; the window size is applied at join time.
  return prior
}

// Keep the current line in full and drop the oldest formatted prior turns
// until the concatenated string fits CONTEXT_MAX_CHARS.
export function joinTurnsKeepingCurrent(formattedTurns: string[], maxChars: number): string {
  // An empty list would only happen if the caller forgot the current DM.
  if (formattedTurns.length === 0) {
    // Return empty so classify() can no-op rather than scoring undefined.
    return ''
  }
  // The last line is always the DM whose banner we may show.
  const current = formattedTurns[formattedTurns.length - 1] ?? ''
  // Copy prior lines so we can shift from the front without mutating the caller.
  const prior = formattedTurns.slice(0, -1)
  // Join helper so the same separator is used on every trim pass.
  const joinAll = (head: string[], tail: string): string =>
    // Skip a leading newline when there is no remaining prior context.
    head.length === 0 ? tail : `${head.join('\n')}\n${tail}`
  // Start from the full window and drop oldest priors while over budget.
  let head = prior
  // Stop when we fit, or when only the current DM remains.
  while (head.length > 0 && joinAll(head, current).length > maxChars) {
    // Drop the oldest formatted turn (left-truncation).
    head = head.slice(1)
  }
  // Current may still exceed maxChars alone; the model tokenizer will truncate.
  return joinAll(head, current)
}

// Build the plaintext string the ONNX classifiers actually see: last 5–8
// verified turns, current DM last, oldest dropped first if over the char budget.
export function buildConversationScoringText(
  transcript: readonly ConversationTurn[],
  currentId: string,
  currentPlaintext: string,
  currentDirection: ConversationTurn['direction'],
  windowSize: number = CONTEXT_WINDOW_MESSAGES,
): string {
  // Pull verified turns that appeared before this DM in the open transcript.
  const prior = collectPriorTurns(transcript, currentId)
  // Keep at most windowSize - 1 priors so the current DM still fits in the window.
  const keptPrior = prior.slice(-(Math.max(1, windowSize) - 1))
  // Format priors with speaker labels.
  const formattedPrior = keptPrior.map((turn) => formatConversationTurn(turn, turn.plaintext ?? ''))
  // Format the DM we are scoring (use the fresh plaintext, not a stale edit).
  const formattedCurrent = formatConversationTurn(
    { id: currentId, direction: currentDirection, plaintext: currentPlaintext, verificationFailed: false },
    currentPlaintext,
  )
  // Concatenate and left-trim to the character budget.
  return joinTurnsKeepingCurrent([...formattedPrior, formattedCurrent], CONTEXT_MAX_CHARS)
}

// Raise the ChatScreen banner cut without rewriting ONNX sidecar thresholds.
// Load-check still uses classifyTfidf/classifyDistilbert/classifyLstm raw.
export function applyChatScreenThreshold(result: ClassifyResult): ClassifyResult {
  // Keep pScam, checkpoint, and timing; only the banner boolean is a product policy.
  return {
    // Copy the probability the graph actually produced.
    ...result,
    // Warn only when the frozen sidecar would warn AND P(scam) clears 0.35.
    warned: result.warned && result.pScam >= CHATSCREEN_MIN_SCAM_THRESHOLD,
  }
}
