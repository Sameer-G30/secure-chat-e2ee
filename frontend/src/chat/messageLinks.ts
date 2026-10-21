// Split verified message text into plain runs and web links.
// https://, http://, www.hosts, and bare hosts such as youtube.com all become anchors.
// The href is always http: or https:. javascript:, data:, and other schemes stay text.
// A bare host is opened as https. This module never fetches a URL and never sends it to the server.

// One slice of a verified bubble, in order.
export interface MessageTextPart {
  // 'link' is a web URL. 'text' is everything else, including rejected schemes.
  kind: 'text' | 'link'
  // Visible characters. For a link this is the sender's spelling, not a rewritten host.
  text: string
  // Absolute http(s) URL for the href. Absent on text runs.
  href?: string
}

// Endings that turn a bare host into a link: .com, .in, and the other public suffixes people paste.
// example.co.in matches because the last label is "in". File suffixes such as .pdf are not in this set.
const BARE_HOST_TLDS: readonly string[] = [
  'com',
  'net',
  'org',
  'io',
  'co',
  'info',
  'biz',
  'xyz',
  'top',
  'click',
  'link',
  'app',
  'dev',
  'edu',
  'gov',
  'me',
  'tv',
  'ly',
  'cc',
  'ai',
  'online',
  'site',
  'tech',
  'store',
  'blog',
  'shop',
  'wiki',
  'news',
  'live',
  'pro',
  'name',
  'mobi',
  'asia',
  'in',
  'uk',
  'us',
  'pk',
  'bd',
  'lk',
  'np',
  'ae',
  'sa',
  'qa',
  'eg',
  'ng',
  'za',
  'au',
  'nz',
  'sg',
  'hk',
  'jp',
  'kr',
  'cn',
  'tw',
  'th',
  'vn',
  'my',
  'id',
  'ph',
  'de',
  'fr',
  'it',
  'es',
  'nl',
  'se',
  'no',
  'dk',
  'fi',
  'pl',
  'cz',
  'at',
  'ch',
  'ie',
  'pt',
  'gr',
  'tr',
  'il',
  'br',
  'mx',
  'ar',
  'cl',
  'pe',
  'co',
  'ru',
  'ua',
  'ro',
  'hu',
  'be',
  'ca',
  'eu',
]

// Longest ending first so the alternation cannot hide a shorter suffix inside a longer one.
const BARE_HOST_TLD_PATTERN = [...BARE_HOST_TLDS].sort((left, right) => right.length - left.length).join('|')

// http(s), a www host, or a bare domain ending. The scheme alternative is first so it wins the whole URL.
const LINK_RE = new RegExp(
  // Explicit http(s) or a www. host, through the next whitespace. Works at the start, middle, or end.
  String.raw`(?:https?:\/\/|www\.)[^\s<>"']+` +
    // Or a bare host that is not the domain half of an email address.
    String.raw`|(?<!@)\b(?:[a-z0-9-]+\.)+(?:${BARE_HOST_TLD_PATTERN})\b(?::\d{2,5})?(?:[/?#][^\s<>"']*)?`,
  'gi',
)

// Sentence punctuation a regex commonly captures after a URL.
const TRAILING_PUNCT_RE = /[.,);\]!?'"]+$/

// Trim trailing punctuation and return how many leading characters belong to the URL.
function linkSpan(raw: string): { href: string; visibleLength: number } | null {
  // Start from the full match.
  let cleaned = raw
  // Peel trailing punctuation the match included from the sentence.
  while (TRAILING_PUNCT_RE.test(cleaned)) {
    // Drop one trailing punctuation run.
    cleaned = cleaned.replace(TRAILING_PUNCT_RE, '')
  }
  // Drop a wrapping ')' that the sender put around the link.
  if (cleaned.endsWith(')') && (cleaned.split('(').length - 1) < (cleaned.split(')').length - 1)) {
    // The closer was not part of the URL.
    cleaned = cleaned.slice(0, -1)
  }
  // An empty remainder is not a link.
  if (!cleaned) {
    // Leave the raw match as text.
    return null
  }
  // Read a scheme when the sender wrote one. Bare hosts have none.
  const scheme = /^[a-z][a-z0-9+.-]*:/i.test(cleaned) ? (cleaned.split(':', 1)[0]?.toLowerCase() ?? '') : ''
  // javascript:, data:, and every other explicit scheme stay text.
  if (scheme !== '' && scheme !== 'http' && scheme !== 'https') {
    // Do not build an href for that scheme.
    return null
  }
  // Bare hosts and www. hosts need a scheme so the href is not a relative path on this origin.
  const absolute = scheme === '' ? `https://${cleaned}` : cleaned
  // Parse locally. This does not connect.
  let parsed: URL
  try {
    // Reject strings the platform does not consider a URL.
    parsed = new URL(absolute)
  } catch {
    // Keep the match as plain text.
    return null
  }
  // The href itself is only http or https, including the https we added for a bare host.
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    // The visible characters stay in the text run.
    return null
  }
  // A host with no dot is not a web address (new URL can still parse some of them).
  if (!parsed.hostname.includes('.')) {
    // Leave it as text.
    return null
  }
  // parsed.href is the absolute URL the browser opens. The visible label stays cleaned.
  return { href: parsed.href, visibleLength: cleaned.length }
}

// Split plaintext into text runs and web links, preserving order.
export function splitHttpLinks(text: string): MessageTextPart[] {
  // No characters, no parts.
  if (!text) {
    // Callers can render nothing.
    return []
  }
  // Accumulate runs in source order.
  const parts: MessageTextPart[] = []
  // Index of the next unmatched character.
  let cursor = 0
  // Walk each web-address match. javascript: and data: are not in the pattern.
  for (const match of text.matchAll(LINK_RE)) {
    // The raw match, including any trailing punctuation.
    const raw = match[0]
    // Where this match starts in the original string.
    const index = match.index ?? 0
    // Accept the match only when it becomes an http(s) href.
    const span = linkSpan(raw)
    // A failed parse stays in the plain-text gap.
    if (!span) {
      // Continue so the next match can still become a link.
      continue
    }
    // Text between the previous link and this one.
    if (index > cursor) {
      // Keep the sender's characters as a text node.
      parts.push({ kind: 'text', text: text.slice(cursor, index) })
    }
    // The link's visible label is the URL without trailing punctuation.
    parts.push({ kind: 'link', text: raw.slice(0, span.visibleLength), href: span.href })
    // Resume after the visible URL so trailing punctuation becomes text.
    cursor = index + span.visibleLength
  }
  // The tail after the last link, or the whole string when there is no link.
  if (cursor < text.length) {
    // One text run for the remainder.
    parts.push({ kind: 'text', text: text.slice(cursor) })
  }
  // Ordered parts for the bubble.
  return parts
}

// True when the bubble contains at least one web link (http, https, www, or a bare host).
export function messageHasHttpLink(text: string): boolean {
  // Scan once. Callers use this to choose the link bubble layout.
  return splitHttpLinks(text).some((part) => part.kind === 'link')
}
