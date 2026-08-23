// Port of ml/src/secure_chat_ml/url_features.py for browser inference.
// Every feature is computed from the URL string itself. This module never
// fetches, never follows a shortener, and never calls a reputation API.

// Frozen shortener hostnames; compared after lowercasing and stripping www.
export const KNOWN_SHORTENER_HOSTS: ReadonlySet<string> = new Set([
  'bit.ly',
  't.co',
  'tinyurl.com',
  'tiny.cc',
  'tiny.one',
  'goo.gl',
  'ow.ly',
  'is.gd',
  'buff.ly',
  'cutt.ly',
  'rebrand.ly',
  'shorturl.at',
  'lnkd.in',
  'db.tt',
  'qr.ae',
  'adf.ly',
  'bit.do',
  't.ly',
  'v.gd',
  'x.co',
  'rb.gy',
  'trib.ly',
  'soo.gd',
  's.id',
  'bl.ink',
])

// Frozen TLDs that are cheap to register or commonly abused in phishing kits.
export const SUSPICIOUS_TLDS: ReadonlySet<string> = new Set([
  'zip',
  'mov',
  'xyz',
  'top',
  'click',
  'link',
  'tk',
  'ml',
  'ga',
  'cf',
  'gq',
  'country',
  'stream',
  'download',
  'racing',
  'review',
  'work',
  'party',
  'science',
  'bid',
  'loan',
  'win',
  'date',
  'faith',
  'cricket',
  'webcam',
  'pw',
  'rest',
  'accountants',
  'support',
  'fit',
  'gdn',
  'quest',
  'cfd',
  'sbs',
  'cyou',
  'icu',
  'cam',
  'hair',
  'makeup',
  'autos',
  'boats',
  'yachts',
  'motorcycles',
  'homes',
  'security',
])

// Path tokens that often mark credential-harvesting landing pages.
const PATH_KEYWORDS: readonly string[] = [
  'login',
  'log-in',
  'signin',
  'sign-in',
  'verify',
  'update',
  'password',
  'passwd',
  'wp-login',
]

// Match http(s) URLs and www. hosts; trailing sentence punctuation is trimmed later.
const SCHEME_OR_WWW_RE = /(?:https?:\/\/|www\.)[^\s<>"')\]]+/gi

// Match schemeless host/path pairs only when a path slash is present (avoids "U.S.").
const SCHEMELESS_RE =
  /(?:[a-z0-9-]+\.)+(?:com|net|org|io|co|info|biz|xyz|top|click|link|ru|cn|uk|us|ly|me|cc|tv|app|dev|edu|gov)\/(?:[^\s<>"')\]]+)/gi

// Pull href='...' / href="..." values out of leftover HTML before tags are stripped.
const HREF_RE = /href\s*=\s*['"]([^'"]+)['"]/gi

// Recognize IPv4 hosts such as 192.0.2.1 without performing any DNS lookup.
const IPV4_RE = /^(?:\d{1,3}\.){3}\d{1,3}$/

// Strip trailing punctuation that URL regexes commonly over-capture from prose.
const TRAILING_PUNCT = /[.,);\]!?'"]+$/

// Schemes that are not fetchable web links and must not enter the feature vector.
const SKIP_SCHEMES: ReadonlySet<string> = new Set([
  'javascript',
  'mailto',
  'tel',
  'sms',
  'data',
  'file',
])

// Ordered names of the aggregated per-message feature vector consumed by sklearn.
export const URL_FEATURE_NAMES: readonly string[] = [
  'has_url',
  'url_count',
  'uses_https',
  'host_is_ip',
  'has_at_sign',
  'num_dots',
  'num_hyphens',
  'num_digits',
  'url_length',
  'path_length',
  'num_subdomains',
  'is_known_shortener',
  'suspicious_tld',
  'path_has_login_verify_update_password_keywords',
  'punycode_xn',
  'digits_in_host',
  'num_dots_mean',
  'num_hyphens_mean',
  'url_length_mean',
  'path_length_mean',
]

// Per-message URL feature map keyed by URL_FEATURE_NAMES.
export type UrlFeatureMap = Record<string, number>

// TRAIN-fitted StandardScaler statistics exported beside each ONNX graph.
export interface UrlScalerSidecar {
  // Ordered feature names that must match URL_FEATURE_NAMES.
  feature_names: string[]
  // TRAIN mean_ vector from sklearn StandardScaler.
  mean: number[]
  // TRAIN scale_ vector (0-std columns already set to 1.0 in sklearn).
  scale: number[]
}

// Trim punctuation and whitespace that is not part of the URL itself.
function trimUrl(raw: string): string {
  // Drop surrounding whitespace copied from the message span.
  let cleaned = raw.trim()
  // Peel trailing sentence punctuation the regex may have included.
  cleaned = cleaned.replace(TRAILING_PUNCT, '')
  // Drop a wrapping closing parenthesis if the opener was not inside the URL.
  if (cleaned.endsWith(')') && (cleaned.split('(').length - 1) < (cleaned.split(')').length - 1)) {
    // Remove the extra closing paren that belonged to surrounding prose.
    cleaned = cleaned.slice(0, -1)
  }
  // Return the cleaned candidate for scheme filtering.
  return cleaned
}

// Decide whether a candidate string is a web URL worth scoring locally.
function isSkippable(raw: string): boolean {
  // Empty strings carry no URL signal.
  if (!raw) {
    // Treat missing candidates as skippable.
    return true
  }
  // Read the scheme up to the first colon, if any.
  const scheme = raw.split(':', 1)[0]?.toLowerCase() ?? ''
  // Skip javascript/mailto/tel/data/file URIs that are not web links.
  return SKIP_SCHEMES.has(scheme)
}

// Find http(s), www, schemeless host/path, and HTML href URLs in one message.
export function extractUrls(text: string): string[] {
  // Treat missing text as having no URLs.
  if (!text) {
    // Return an empty list so callers can iterate uniformly.
    return []
  }
  // Accumulate unique URLs while preserving first-seen order.
  const found: string[] = []
  // Track already-emitted strings so duplicates from href + visible text collapse.
  const seen = new Set<string>()

  // Add one cleaned candidate if it is a real web URL and not a duplicate.
  function add(candidate: string): void {
    // Normalize wrapping punctuation before the uniqueness check.
    const cleaned = trimUrl(candidate)
    // Ignore empty, non-web, or already-recorded candidates.
    if (isSkippable(cleaned) || seen.has(cleaned)) {
      // Skip this candidate without mutating the output list.
      return
    }
    // Remember the candidate so later duplicate spans are ignored.
    seen.add(cleaned)
    // Append in first-seen order for stable feature aggregation.
    found.push(cleaned)
  }

  // Prefer explicit href targets so leftover HTML still yields the real link.
  for (const match of text.matchAll(HREF_RE)) {
    // Record the href value as a URL candidate.
    add(match[1] ?? '')
  }
  // Collect canonical http(s) and www. spans from visible text.
  for (const match of text.matchAll(SCHEME_OR_WWW_RE)) {
    // Record the matched http(s)/www span.
    add(match[0])
  }
  // Collect regex-safe schemeless host/path pairs such as bit.ly/abc.
  for (const match of text.matchAll(SCHEMELESS_RE)) {
    // Record the schemeless candidate; parsing will add a default scheme.
    add(match[0])
  }
  // Return the ordered unique URL list for feature extraction.
  return found
}

// Give URL parsing a scheme so www. and schemeless hosts populate hostname/path.
function forParse(raw: string): string {
  // Leave already-schemed URLs unchanged.
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(raw)) {
    // Return the original schemed URL for parsing.
    return raw
  }
  // Pretend www./host/path strings are http so hostname is populated, not path.
  return `http://${raw}`
}

// Lowercase a hostname and strip a leading www. for list lookups.
function normalizeHost(host: string): string {
  // Lowercase so list membership is case-insensitive.
  const lowered = host.toLowerCase()
  // Strip a leading www. which is not part of shortener or TLD identity.
  if (lowered.startsWith('www.')) {
    // Drop the www. prefix for frozen-list comparisons.
    return lowered.slice(4)
  }
  // Return the already-normalized host.
  return lowered
}

// Detect IPv4 or IPv6 hosts without DNS or socket calls.
function hostIsIp(host: string): boolean {
  // Empty hosts cannot be IP literals.
  if (!host) {
    // Treat missing hosts as not-IP.
    return false
  }
  // Match dotted-decimal IPv4 used in many phishing kits.
  if (IPV4_RE.test(host)) {
    // An IPv4 literal is always scored as host_is_ip.
    return true
  }
  // IPv6 literals contain colons; parsers may wrap them in brackets.
  const stripped = host.replace(/^\[/, '').replace(/\]$/, '')
  // A colon in the stripped host is the cheap IPv6 heuristic (no network).
  return stripped.includes(':')
}

// Count labels to the left of the registrable domain; IP hosts have zero.
function subdomainCount(host: string): number {
  // IP literals have no subdomain structure.
  if (!host || hostIsIp(host)) {
    // Report zero subdomains for IPs and empty hosts.
    return 0
  }
  // Split the normalized host on dots to count labels.
  const labels = normalizeHost(host).split('.').filter(Boolean)
  // Need at least domain + tld before anything counts as a subdomain.
  if (labels.length <= 2) {
    // example.com has zero subdomains.
    return 0
  }
  // Treat every extra label as a subdomain (www.a.b.example.com → 3).
  return labels.length - 2
}

// Read the rightmost label as a TLD heuristic (no Public Suffix List fetch).
function tld(host: string): string {
  // IP literals have no TLD.
  if (!host || hostIsIp(host)) {
    // Report an empty TLD so suspicious_tld stays 0.
    return ''
  }
  // Split on dots and take the last non-empty label.
  const labels = host.toLowerCase().split('.').filter(Boolean)
  // Return the last label, or empty if the host was only dots.
  return labels.length > 0 ? (labels[labels.length - 1] ?? '') : ''
}

// Split a URL locally, falling back to string heuristics when URL() rejects it.
function splitUrl(url: string): [string, string, string, string] {
  // Prefer the platform parser when the string is a well-formed URL.
  try {
    // Normalize schemeless hosts so hostname is populated.
    const parsed = new URL(forParse(url))
    // protocol includes the colon; strip it to match Python's scheme field.
    const scheme = parsed.protocol.replace(/:$/, '')
    // hostname is already unbracketed for IPv6.
    const host = parsed.hostname || ''
    // pathname includes the leading slash when present.
    const path = parsed.pathname || ''
    // search includes the leading '?'; drop it to match urlparse.query.
    const query = parsed.search.startsWith('?') ? parsed.search.slice(1) : parsed.search
    // Return the four fields used by the lexical feature function.
    return [scheme, host, path, query]
  } catch {
    // Fall through to linear string heuristics for malformed URLs.
  }
  // Work on a scheme-normalized copy even in the fallback path.
  const raw = forParse(url)
  // Detect https vs http from the prefix only.
  const lowered = raw.toLowerCase()
  // Record the scheme token used by uses_https.
  const scheme = lowered.startsWith('https://') ? 'https' : 'http'
  // Drop the scheme prefix when present.
  const rest = raw.includes('://') ? (raw.split('://')[1] ?? raw) : raw
  // Keep the last @-segment as the host side of userinfo@host.
  const afterAt = rest.includes('@') ? (rest.split('@').pop() ?? rest) : rest
  // Split host from path at the first slash.
  const slash = afterAt.indexOf('/')
  // Host is everything before the first slash (or the whole rest).
  let hostPart = slash === -1 ? afterAt : afterAt.slice(0, slash)
  // Path+query is everything after the first slash.
  const pathAndQuery = slash === -1 ? '' : afterAt.slice(slash + 1)
  // Drop a trailing port when it is a single :digits suffix, not IPv6.
  if ((hostPart.split(':').length - 1) === 1 && !hostPart.startsWith('[')) {
    // Keep only the hostname to the left of the port.
    hostPart = hostPart.split(':')[0] ?? hostPart
  }
  // Split path and query at the first '?'.
  const qIndex = pathAndQuery.indexOf('?')
  // Path has no leading slash in this fallback until we add it.
  const pathBody = qIndex === -1 ? pathAndQuery : pathAndQuery.slice(0, qIndex)
  // Query is everything after '?', minus a leaked fragment.
  const queryRaw = qIndex === -1 ? '' : pathAndQuery.slice(qIndex + 1)
  // Drop a fragment if one leaked into the query string.
  const query = queryRaw.split('#')[0] ?? ''
  // Return heuristic fields; host may still be an opaque malformed literal.
  return [scheme, hostPart.replace(/^\[/, '').replace(/\]$/, ''), pathBody ? `/${pathBody}` : '', query]
}

// Score one URL into the lexical/structural features listed in the spec.
export function extractSingleUrlFeatures(url: string): Record<string, number> {
  // Split scheme/host/path/query locally, tolerating malformed phishing URLs.
  const [scheme, host, path, query] = splitUrl(url)
  // Combine path and query so keyword checks see /login?next=bank as well.
  const pathAndQuery = path + (query ? `?${query}` : '')
  // Normalize the host for shortener and TLD lookups.
  const hostNorm = normalizeHost(host)
  // Count dots in the hostname (classic phishing visual-confusion feature).
  const numDots = (host.match(/\./g) ?? []).length
  // Count hyphens in the hostname (brand-impersonation feature).
  const numHyphens = (host.match(/-/g) ?? []).length
  // Count digit characters in the full original URL string.
  const numDigits = (url.match(/\d/g) ?? []).length
  // Count digit characters in the hostname (paypa1.com, 192.0.2.1).
  const digitsInHost = (host.match(/\d/g) ?? []).length
  // Measure the raw URL length including scheme and query.
  const urlLength = url.length
  // Measure the path length excluding the host.
  const pathLength = path.length
  // Flag https; http-only and schemeless (treated as http) score 0.
  const usesHttps = scheme.toLowerCase() === 'https' ? 1 : 0
  // Flag IP-literal hosts used to skip brand-looking domains.
  const hostIsIpFlag = hostIsIp(host) ? 1 : 0
  // Flag '@' which can hide the real host in http://user@evil/ paths.
  const hasAtSign = url.includes('@') ? 1 : 0
  // Flag frozen shortener hosts; this is a list lookup, not a resolve.
  const isKnownShortener =
    KNOWN_SHORTENER_HOSTS.has(hostNorm) ||
    [...KNOWN_SHORTENER_HOSTS].some((name) => hostNorm.endsWith(`.${name}`))
      ? 1
      : 0
  // Flag frozen suspicious TLDs using the last hostname label only.
  const suspiciousTld = SUSPICIOUS_TLDS.has(tld(host)) ? 1 : 0
  // Flag credential-harvesting keywords anywhere in path or query.
  const loweredPath = pathAndQuery.toLowerCase()
  // Score 1 when any configured path keyword is a substring of path/query.
  const pathKeywords = PATH_KEYWORDS.some((keyword) => loweredPath.includes(keyword)) ? 1 : 0
  // Flag punycode labels used in IDN homograph lookalikes.
  const punycode = host.toLowerCase().includes('xn--') ? 1 : 0
  // Count subdomain labels left of domain.tld.
  const numSubdomains = subdomainCount(host)
  // Return a dict so aggregators can max/mean/any by name.
  return {
    uses_https: usesHttps,
    host_is_ip: hostIsIpFlag,
    has_at_sign: hasAtSign,
    num_dots: numDots,
    num_hyphens: numHyphens,
    num_digits: numDigits,
    url_length: urlLength,
    path_length: pathLength,
    num_subdomains: numSubdomains,
    is_known_shortener: isKnownShortener,
    suspicious_tld: suspiciousTld,
    path_has_login_verify_update_password_keywords: pathKeywords,
    punycode_xn: punycode,
    digits_in_host: digitsInHost,
  }
}

// Collapse every URL in a message into one fixed-length vector (max/mean/any).
export function extractMessageUrlFeatures(text: string): UrlFeatureMap {
  // Collect URL strings from the message without touching the network.
  const urls = extractUrls(text)
  // Build the zero vector used when the message has no URLs at all.
  const zeros: UrlFeatureMap = {}
  // Initialize every documented name to 0.0.
  for (const name of URL_FEATURE_NAMES) {
    // Zero is the legitimate link-free default, not a missing value.
    zeros[name] = 0
  }
  // Short-circuit so legitimate link-free DMs do not look "missing".
  if (urls.length === 0) {
    // has_url stays 0 and every other feature stays 0.
    return zeros
  }
  // Score each URL independently before aggregating.
  const perUrl = urls.map((url) => extractSingleUrlFeatures(url))
  // Binary/any features: a single suspicious URL should surface.
  const anyNames = [
    'uses_https',
    'host_is_ip',
    'has_at_sign',
    'is_known_shortener',
    'suspicious_tld',
    'path_has_login_verify_update_password_keywords',
    'punycode_xn',
  ] as const
  // Continuous features: keep the max (worst/longest) plus a mean copy.
  const maxMeanNames = ['num_dots', 'num_hyphens', 'num_digits', 'url_length', 'path_length'] as const
  // Start from zeros and overwrite with aggregates.
  const aggregated: UrlFeatureMap = { ...zeros }
  // Record that at least one URL was present.
  aggregated.has_url = 1
  // Record how many unique URLs the message contained.
  aggregated.url_count = urls.length
  // Reduce binary flags with max, which is equivalent to logical any.
  for (const name of anyNames) {
    // Take the strongest (1) observation across URLs.
    aggregated[name] = Math.max(...perUrl.map((row) => row[name] ?? 0))
  }
  // Reduce continuous features with both max and mean as the spec allows.
  for (const name of maxMeanNames) {
    // Collect the per-URL values for this continuous feature.
    const values = perUrl.map((row) => row[name] ?? 0)
    // Store the max under the canonical spec name (num_dots, url_length, ...).
    aggregated[name] = Math.max(...values)
    // Store the mean under an explicit _mean suffix for the extra columns.
    aggregated[`${name}_mean`] = values.reduce((sum, value) => sum + value, 0) / values.length
  }
  // num_subdomains and digits_in_host are max-aggregated (worst host wins).
  aggregated.num_subdomains = Math.max(...perUrl.map((row) => row.num_subdomains ?? 0))
  // digits_in_host uses max so an IP host is not averaged away by a second URL.
  aggregated.digits_in_host = Math.max(...perUrl.map((row) => row.digits_in_host ?? 0))
  // Return the fixed-key dict matching URL_FEATURE_NAMES.
  return aggregated
}

// Turn aggregated dicts into the column order sklearn will train on.
export function featuresToVector(features: UrlFeatureMap): number[] {
  // Look up each name so the vector layout cannot drift from the name tuple.
  return URL_FEATURE_NAMES.map((name) => Number(features[name] ?? 0))
}

// Apply a TRAIN-fitted StandardScaler: (x - mean) / scale.
export function scaleUrlFeatures(raw: number[], scaler: UrlScalerSidecar): Float32Array {
  // Guard a width mismatch so a stale sidecar cannot silently scramble the head.
  if (raw.length !== scaler.mean.length || raw.length !== scaler.scale.length) {
    // Fail rather than feeding the logistic/LSTM head a shifted vector.
    throw new Error('URL scaler width does not match the 20-d feature vector')
  }
  // Allocate the scaled row the ONNX URL concat / logistic block expects.
  const scaled = new Float32Array(raw.length)
  // Z-score each coordinate independently.
  for (let index = 0; index < raw.length; index += 1) {
    // sklearn sets 0-std columns' scale_ to 1.0, so division is always safe.
    const scale = scaler.scale[index] ?? 1
    // Subtract the TRAIN mean then divide by TRAIN scale.
    scaled[index] = (raw[index] - (scaler.mean[index] ?? 0)) / (scale === 0 ? 1 : scale)
  }
  // Return float32 to match the ONNX graph input.
  return scaled
}

// Extract, vectorize, and z-score URL features for one chat message.
export function scaledUrlFeatureVector(text: string, scaler: UrlScalerSidecar): Float32Array {
  // Messages with no URL still get a zero raw vector (has_url=0), then scaled.
  const raw = featuresToVector(extractMessageUrlFeatures(text))
  // Apply TRAIN mean/scale; do not recompute statistics in the browser.
  return scaleUrlFeatures(raw, scaler)
}
