// ChatScreen scoring policy: a short conversation window, a skip for trivial
// URL-free lines, and a slightly higher banner threshold than the frozen VAL cut.
// Ciphertext never enters this module; callers pass already-verified plaintext.

// Reuse the on-device URL extractor so a short "t.co/..." DM is never skipped.
// Feature and shortener checks stay local: this module never fetches a URL.
import { extractSingleUrlFeatures, extractUrls, KNOWN_SHORTENER_HOSTS } from './urlFeatures'
// Reuse ClassifyResult so the ChatScreen overlay can flip `warned` without a second type.
import type { ClassifyResult } from './types'

// Number of recent verified turns (including the current DM) fed to the model.
// Three keeps the previous exchange plus this message, so an older turn cannot
// pull a normal reply over the banner line.
export const CONTEXT_WINDOW_MESSAGES = 3

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

// Build the plaintext string the ONNX classifiers actually see: last 3
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

// Prose outside one URL may be this long and still count as a short benign link line.
// "here is the shared doc" fits; a paragraph does not, and stays on the normal window score.
export const SHORT_BENIGN_LINK_PROSE_CHARS = 120

// Hosts whose lone http(s) link is not treated as a phishing signal by itself.
// Matching is exact or a subdomain. Nothing here is fetched or resolved.
const WELL_KNOWN_LINK_HOSTS: ReadonlySet<string> = new Set([
  // The hosts called out for shared-doc false positives.
  'docs.google.com',
  'drive.google.com',
  'github.com',
  'youtube.com',
  // Parent and sibling hosts of the same kind (docs, code, video, files).
  'google.com',
  'youtu.be',
  'gitlab.com',
  'microsoft.com',
  'office.com',
  'office365.com',
  'live.com',
  'sharepoint.com',
  'dropbox.com',
  'notion.so',
  'notion.site',
  'wikipedia.org',
  'apple.com',
  'icloud.com',
  'zoom.us',
  'slack.com',
  'figma.com',
  'stackoverflow.com',
  'reddit.com',
  'linkedin.com',
  'medium.com',
  'mozilla.org',
])

// Lowercase a hostname and drop a single leading www. for list checks.
function normalizeLinkHost(hostname: string): string {
  // Host comparisons are case-insensitive.
  const lowered = hostname.toLowerCase().replace(/\.$/, '')
  // www.youtube.com is the same site as youtube.com for this allow-list.
  return lowered.startsWith('www.') ? lowered.slice(4) : lowered
}

// True when the hostname is a listed host or a subdomain of one.
function isWellKnownLinkHost(hostname: string): boolean {
  // Empty hosts cannot be a shared-doc site.
  const host = normalizeLinkHost(hostname)
  // Walk the frozen list. No DNS lookup.
  for (const known of WELL_KNOWN_LINK_HOSTS) {
    // Exact host or a real subdomain (docs.google.com.evil.com does not match).
    if (host === known || host.endsWith(`.${known}`)) {
      // This URL's host is on the benign list.
      return true
    }
  }
  // Unknown hosts are not excused by the short-line rule.
  return false
}

// Parse a URL string locally and return its http(s) hostname, or null.
function httpUrlHostname(url: string): string | null {
  // URL() throws on schemeless tokens such as bit.ly/abc.
  try {
    // Accept only strings that already have a scheme.
    const parsed = new URL(url)
    // javascript:, data:, and mailto: are not the links this policy excuses.
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      // Leave those strings for the model and the renderer to ignore.
      return null
    }
    // hostname has no port and no userinfo.
    return parsed.hostname
  } catch {
    // Malformed candidates are not http(s) links.
    return null
  }
}

// Hostname for shortener checks, including schemeless extractor hits.
function hostForListLookup(url: string): string {
  // Prefer the real hostname when the string already has a scheme.
  const direct = httpUrlHostname(url)
  // http(s) URLs use the platform hostname.
  if (direct) {
    // Normalize www. before the frozen shortener set.
    return normalizeLinkHost(direct)
  }
  // Schemeless hits still need a host for the bit.ly check. This does not connect.
  try {
    // Pretend http so URL() fills hostname. The scheme is discarded.
    return normalizeLinkHost(new URL(`http://${url}`).hostname)
  } catch {
    // Give up rather than guessing.
    return ''
  }
}

// True when the host is a frozen shortener or a subdomain of one.
function hostIsShortener(hostname: string): boolean {
  // Compare the normalized host to the training shortener list.
  const host = normalizeLinkHost(hostname)
  // Exact membership covers bit.ly, t.co, and the rest of the frozen set.
  if (KNOWN_SHORTENER_HOSTS.has(host)) {
    // Shorteners stay a scam pattern even when the rest of the line is short.
    return true
  }
  // A subdomain of a shortener is the same pattern.
  for (const name of KNOWN_SHORTENER_HOSTS) {
    // Suffix match is label-bounded because of the leading dot.
    if (host.endsWith(`.${name}`)) {
      // Treat the subdomain as the shortener.
      return true
    }
  }
  // Ordinary hosts are not shorteners.
  return false
}

// True when this URL has a lexical phishing feature other than "it is a URL".
function urlIsLexicallySuspicious(url: string): boolean {
  // Score the string the extractor already accepted. No network.
  const features = extractSingleUrlFeatures(url.includes('://') ? url : `http://${url}`)
  // Shorteners, odd TLDs, IP hosts, userinfo, punycode, and login paths are extra signals.
  if (features.is_known_shortener === 1) {
    // bit.ly and the rest of the frozen list.
    return true
  }
  // Cheap or abused TLDs from the training feature list.
  if (features.suspicious_tld === 1) {
    // A .zip / .tk host is not "just a link".
    return true
  }
  // Raw IP hosts skip brand-looking domains.
  if (features.host_is_ip === 1) {
    // Keep the model banner.
    return true
  }
  // user@host hides the real destination.
  if (features.has_at_sign === 1) {
    // Keep the model banner.
    return true
  }
  // xn-- labels are homograph candidates.
  if (features.punycode_xn === 1) {
    // Keep the model banner.
    return true
  }
  // /login, /verify, /password in the path or query.
  if (features.path_has_login_verify_update_password_keywords === 1) {
    // The path is a second suspicious feature, so the URL is not excused.
    return true
  }
  // The feature helper and the host list agree on shorteners.
  return hostIsShortener(hostForListLookup(url))
}

// Phrases the chat policy treats as scam context around an otherwise ordinary link.
function textHasScamPhrase(text: string): boolean {
  // Match on a lowercased copy so "Verify Immediately" still counts.
  const lowered = text.toLowerCase()
  // Gift-card payment requests.
  if (lowered.includes('gift card')) {
    // "gift cards" contains this substring.
    return true
  }
  // The urgency line called out for shortener scams.
  if (lowered.includes('verify immediately')) {
    // Keep the banner for this phrase.
    return true
  }
  // "account locked", "account is locked", and "account has been locked".
  return /account(?:\s+\S+){0,4}\s+locked\b/.test(lowered)
}

// Text left after removing one extracted URL, with surrounding punctuation collapsed.
function proseOutsideUrl(text: string, url: string): string {
  // The extractor's cleaned URL should be a substring of the bubble.
  const index = text.indexOf(url)
  // If it is not, fail closed: the whole bubble counts as prose.
  if (index < 0) {
    // A failed split must not look like a URL-only message.
    return text.trim()
  }
  // Drop that one occurrence and keep whatever the sender wrote around it.
  const raw = `${text.slice(0, index)} ${text.slice(index + url.length)}`
  // Collapse leftover brackets and sentence punctuation so "see (url)." has prose "see".
  return raw.replace(/[\s.,;:!?()[\]"'<>]+/g, ' ').trim()
}

// True when this bubble is only a URL, or a short line whose only extra feature is one
// http(s) URL on a well-known host. The model still scores the window; this only
// decides whether that URL is allowed to raise the banner by itself.
export function isBenignLinkTurn(text: string): boolean {
  // Work from trimmed plaintext. The server never sees this string.
  const trimmed = text.trim()
  // Empty bubbles are handled by the trivial-skip path.
  if (!trimmed) {
    // Nothing to excuse.
    return false
  }
  // Every URL-like token, including schemeless shorteners.
  const found = extractUrls(trimmed)
  // http(s) only. javascript: and data: never become a benign link turn.
  const httpUrls = found.filter((url) => httpUrlHostname(url) !== null)
  // A schemeless copy of the same http(s) URL is one link, not a second destination.
  const otherUrls = found.filter(
    (url) => httpUrlHostname(url) === null && !httpUrls.some((httpUrl) => httpUrl.includes(url)),
  )
  // One http(s) URL and no other destination (a second bit.ly stays suspicious).
  if (httpUrls.length !== 1 || otherUrls.length !== 0) {
    // Normal text, including a long note that happens to contain a link, keeps the model score.
    return false
  }
  // The single http(s) URL.
  const url = httpUrls[0] ?? ''
  // Hostname for the well-known list.
  const host = httpUrlHostname(url) ?? ''
  // Words other than that URL.
  const prose = proseOutsideUrl(trimmed, url)
  // A bubble that is only the URL: do not banner from the URL alone when it has no other bad feature.
  if (prose.length === 0) {
    // Shorteners, login paths, and lookalike hosts stay on the model path.
    return !urlIsLexicallySuspicious(url)
  }
  // Longer messages are normal text. The three-turn window score stands.
  if (prose.length > SHORT_BENIGN_LINK_PROSE_CHARS) {
    // A paragraph with a docs link is not the short-line exception.
    return false
  }
  // "verify immediately" in the same bubble is context, not a bare link.
  if (textHasScamPhrase(trimmed)) {
    // Leave the banner decision to the model and the 0.35 overlay.
    return false
  }
  // The short-line exception is only for well-known hosts.
  if (!isWellKnownLinkHost(host)) {
    // An unknown host plus prose is scored as normal text.
    return false
  }
  // A login path or shortener on an otherwise familiar host is a second feature.
  return !urlIsLexicallySuspicious(url)
}

// True when the scored window (prior turns plus this bubble) contains a scam pattern.
export function windowHasScamPattern(windowText: string): boolean {
  // Account-locked, gift-card, and verify-immediately lines anywhere in the window.
  if (textHasScamPhrase(windowText)) {
    // That context is enough to keep a banner on a following docs link.
    return true
  }
  // Shorteners in this bubble or an earlier turn are a scam pattern of their own.
  for (const url of extractUrls(windowText)) {
    // Host check only. The URL is not opened.
    if (hostIsShortener(hostForListLookup(url))) {
      // bit.ly and the rest of the frozen list keep the banner.
      return true
    }
  }
  // No listed pattern. A well-known link can be excused.
  return false
}

// Drop a banner that the URL itself produced, unless the window around it is a scam pattern.
// Frozen sidecar thresholds and CHATSCREEN_MIN_SCAM_THRESHOLD are unchanged.
export function applyBenignLinkBannerPolicy(
  result: ClassifyResult,
  currentPlaintext: string,
  windowText: string,
): ClassifyResult {
  // A score under the banner cut stays quiet.
  if (!result.warned) {
    // Copy the result through so callers can always use the return value.
    return result
  }
  // Normal text keeps the three-turn score, including its banner.
  if (!isBenignLinkTurn(currentPlaintext)) {
    // Verify-immediately and shortener bubbles land here.
    return result
  }
  // A benign link after "account locked" / gift cards / a shortener still banners.
  if (windowHasScamPattern(windowText)) {
    // The surrounding window is the reason, not the well-known URL alone.
    return result
  }
  // Same probability and checkpoint. Only the product banner bit changes.
  return {
    // Keep the graph's probability.
    ...result,
    // Do not raise the banner from this URL alone.
    warned: false,
  }
}
