"""Deterministic email/SMS → WhatsApp/DM-style rewrite (no LLM, no network).

`rewrite_method` is always `rule_based_v1`. Labels are never changed. URLs
found in the original text are kept in the rewritten text. Legitimate rows
do not gain urgency or phishing boilerplate. This module must not read or
write `data/chat_eval/`; the locked chat-style eval set is never rewritten
into training.
"""

# Import re to strip email chrome, split sentences, and apply contractions.
import re

# Import URL extraction so links survive stripping and can be re-attached.
from secure_chat_ml.url_features import extract_urls

# Documented rewrite identifier stored on every processed_chat row.
REWRITE_METHOD = "rule_based_v1"

# Hard cap so TF-IDF sees chat-scale text rather than full email threads.
MAX_REWRITE_CHARS = 400

# Placeholder prefix used to protect URLs while contractions run on the body.
_URL_PLACEHOLDER_PREFIX = "URLTOKEN"
_URL_PLACEHOLDER_SUFFIX = "XYZ"

# Binary labels matching data/label-schema.yaml (never flipped by a rewrite).
LEGITIMATE_LABEL = 0
SCAM_LABEL = 1

# Verbs and frames that must not be inserted into legitimate rewrites.
_HAM_FORBIDDEN_PHRASES = (
    "verify your",
    "urgent",
    "suspended",
    "gift card",
    "wire money",
    "act now",
    "you have won",
    "claim your",
    "password now",
    "otp",
    "seed phrase",
)

# Casual openers used only on scam rows that still read like leftover email.
_SCAM_OPENERS = ("hey, ", "yo, ", "hi, ", "pls read — ")

# Ordered (pattern, replacement) contractions applied outside URL placeholders.
_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[Yy]ou are\b"), "you're"),
    (re.compile(r"\b[Ww]e are\b"), "we're"),
    (re.compile(r"\b[Tt]hey are\b"), "they're"),
    (re.compile(r"\b[Ii]t is\b"), "it's"),
    (re.compile(r"\b[Tt]hat is\b"), "that's"),
    (re.compile(r"\b[Dd]o not\b"), "don't"),
    (re.compile(r"\b[Dd]oes not\b"), "doesn't"),
    (re.compile(r"\b[Dd]id not\b"), "didn't"),
    (re.compile(r"\b[Ww]ill not\b"), "won't"),
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can't"),
    (re.compile(r"\bcan not\b", re.IGNORECASE), "can't"),
    (re.compile(r"\b[Ii] am\b"), "I'm"),
    (re.compile(r"\b[Pp]lease\b"), "pls"),
    (re.compile(r"\b[Tt]hank you\b"), "thanks"),
)

# Header field names dropped when they appear as the first token on a line.
_HEADER_PREFIXES = (
    "from:",
    "to:",
    "cc:",
    "bcc:",
    "subject:",
    "date:",
    "sent:",
    "reply-to:",
    "return-path:",
    "message-id:",
    "content-type:",
    "mime-version:",
)

# Cut quoted reply threads at the first of these case-insensitive substrings.
_THREAD_MARKERS = (
    "forwarded by",
    "original message",
    "begin forwarded message",
    "-----original message-----",
)

# Cut confidentiality / legal disclaimers from the first of these substrings.
_DISCLAIMER_MARKERS = (
    "this email is confidential",
    "this e-mail is confidential",
    "this message is confidential",
    "this communication is confidential",
    "if you are not the intended recipient",
    "privileged and confidential",
)

# Line-level tokens that mark mailing-list chrome rather than message meaning.
_UNSUBSCRIBE_TOKENS = ("unsubscribe", "opt-out", "opt out", "manage your preferences")

# Strip tags linearly after hrefs have already been harvested (no HTMLParser).
_TAG_RE = re.compile(r"<[^>]+>")

# Drop formal openings that never appear in a DM.
_GREETING_RE = re.compile(
    r"^(?:dear\s+[\w .'-]+[,:]?\s*|hello(?:\s+\w+)?[,!]?\s*|"
    r"hi\s+(?:there|sir|madam|team|customer|user|all|valued)[,!]?\s*|"
    r"greetings(?:\s+\w+)?[,:]?\s*|"
    r"good\s+(?:morning|afternoon|evening)(?:\s+\w+)?[,!]?\s*|"
    r"to\s+whom\s+it\s+may\s+concern[,:]?\s*)",
    re.IGNORECASE,
)

# Drop formal closings and phone-signature leftovers.
_CLOSING_RE = re.compile(
    r"(?:best\s+regards|kind\s+regards|warm\s+regards|sincerely|"
    r"yours\s+(?:truly|faithfully)|thanks\s+and\s+regards|"
    r"sent from my (?:iphone|ipad|android|samsung)|"
    r"get outlook for (?:ios|android)).*$",
    re.IGNORECASE | re.DOTALL,
)

# Collapse runs of whitespace after stripping artifacts.
_WHITESPACE_RE = re.compile(r"\s+")

# Detect leftover HTML so it can be flattened without dropping href URLs.
_HTML_RE = re.compile(r"(?i)<(?:html|body|div|p|br|a|table|span|font)\b")


# Protect URLs with placeholders so contractions cannot rewrite inside them.
def _protect_urls(text: str, urls: list[str]) -> tuple[str, dict[str, str]]:
    """Return (text_with_placeholders, placeholder→url) for reversible edits."""

    # Map each placeholder token back to its original URL string.
    mapping: dict[str, str] = {}
    # Start from the original text and replace longer URLs first to avoid partial hits.
    protected = text
    # Sort by length descending so a prefix URL cannot swallow a longer one.
    for index, url in enumerate(sorted(urls, key=len, reverse=True)):
        # Build a token that contractions and regexes will not split.
        token = f"{_URL_PLACEHOLDER_PREFIX}{index}{_URL_PLACEHOLDER_SUFFIX}"
        # Remember how to restore this URL after body edits.
        mapping[token] = url
        # Replace every remaining occurrence of this URL with the token.
        protected = protected.replace(url, token)
    # Return the protected body and the restore map.
    return protected, mapping


# Put original URL strings back after contractions and truncation.
def _restore_urls(text: str, mapping: dict[str, str]) -> str:
    """Replace placeholder tokens with the original URL strings."""

    # Start from the possibly edited body.
    restored = text
    # Restore each token; missing tokens mean truncation dropped that URL.
    for token, url in mapping.items():
        # Substitute the original URL back wherever the token survived.
        restored = restored.replace(token, url)
    # Return the body with real URLs restored.
    return restored


# Cut text at the earliest case-insensitive marker; linear in len(text).
def _cut_at_first_marker(text: str, markers: tuple[str, ...]) -> str:
    """Return the prefix before the first marker, or the full text if none match."""

    # Search a lowercase copy so marker matching is case-insensitive.
    lowered = text.lower()
    # Start with "no cut" equal to the full length.
    cut = len(text)
    # Take the earliest marker so quoted threads and disclaimers are dropped.
    for marker in markers:
        # Find is linear and does not backtrack.
        index = lowered.find(marker)
        # Ignore missing markers; keep the leftmost hit.
        if 0 <= index < cut:
            # Move the cut to this earlier marker.
            cut = index
    # Return the surviving prefix, stripped of trailing whitespace.
    return text[:cut].strip()


# Flatten HTML to text after hrefs have already been harvested by extract_urls.
def _flatten_html(text: str) -> str:
    """Return plain text when the body still looks like HTML."""

    # Skip tag stripping when there is no markup to flatten.
    if not _HTML_RE.search(text):
        # Return the original body unchanged.
        return text
    # Replace tags with spaces in linear time (hrefs were already extracted).
    flattened = _TAG_RE.sub(" ", text)
    # Fall back to the original if flattening produced nothing useful.
    return flattened.strip() or text


# Drop header lines, quoted-reply lines, and unsubscribe chrome in one pass.
def _filter_email_lines(text: str) -> str:
    """Return text with RFC822, quote, and unsubscribe lines removed."""

    # Accumulate surviving lines for a single join at the end.
    kept: list[str] = []
    # Walk lines once so the filter stays linear in the body length.
    for raw_line in text.splitlines():
        # Strip edges for prefix checks while keeping inner wording.
        line = raw_line.strip()
        # Skip empty lines; whitespace is collapsed later anyway.
        if not line:
            # Continue with the next source line.
            continue
        # Skip quoted-reply lines that start with '>'.
        if line.startswith(">"):
            # Drop the quoted line without copying it.
            continue
        # Lowercase once for header and unsubscribe checks.
        lowered = line.lower()
        # Skip leftover RFC822 header lines.
        if lowered.startswith(_HEADER_PREFIXES):
            # Header chrome is not chat-register content.
            continue
        # Skip X-* header lines (require a colon so "x-ray" prose is kept).
        if lowered.startswith("x-") and ":" in line[:48]:
            # Drop the X-header line.
            continue
        # Skip mailing-list unsubscribe / preference-management lines.
        if any(token in lowered for token in _UNSUBSCRIBE_TOKENS):
            # Drop the chrome line.
            continue
        # Keep this line as candidate chat text.
        kept.append(line)
    # Join surviving lines with spaces so later sentence splits see one paragraph.
    return " ".join(kept)


# Remove headers, quoted threads, unsubscribes, and disclaimers.
def strip_email_artifacts(text: str) -> str:
    """Return the body with remaining email chrome removed, before shortening."""

    # Flatten leftover HTML so later filters see visible text, not tags.
    body = _flatten_html(text)
    # Cut at the first forwarded/original-message marker so replies are not kept.
    body = _cut_at_first_marker(body, _THREAD_MARKERS)
    # Drop header, quote, and unsubscribe lines in a single linear pass.
    body = _filter_email_lines(body)
    # Cut confidentiality disclaimers from the first match through the end.
    body = _cut_at_first_marker(body, _DISCLAIMER_MARKERS)
    # Drop a formal closing if one remains at the end of the body.
    body = _CLOSING_RE.split(body, maxsplit=1)[0]
    # Collapse whitespace so later sentence splits see a single paragraph.
    body = _WHITESPACE_RE.sub(" ", body).strip()
    # Drop a leading Dear/Hello/Good morning greeting once.
    body = _GREETING_RE.sub("", body).strip()
    # Return the cleaned body, which may still be empty for disclaimer-only mail.
    return body


# Apply contractions and light chat punctuation outside protected URLs.
def _informalize(text: str) -> str:
    """Return a slightly more DM-like version of the cleaned body."""

    # Start from the cleaned body.
    informal = text
    # Apply each contraction pattern in a deterministic order.
    for pattern, replacement in _CONTRACTIONS:
        # Replace formal phrases with contractions / pls / thanks.
        informal = pattern.sub(replacement, informal)
    # Collapse leftover spaces after substitutions.
    informal = _WHITESPACE_RE.sub(" ", informal).strip()
    # Return the informalized body.
    return informal


# Keep the first 1–3 sentences so the rewrite stays a short DM.
def _first_sentences(text: str, max_sentences: int = 3) -> str:
    """Return up to max_sentences sentences from the start of the body."""

    # If the body is already short, keep it as a single DM line.
    if len(text) <= 160:
        # Short SMS-like text does not need sentence chopping.
        return text
    # Split on sentence-ending punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Keep only the leading sentences.
    kept = [part.strip() for part in parts[:max_sentences] if part.strip()]
    # Join the kept sentences with a single space.
    clipped = " ".join(kept).strip()
    # Fall back to the original if splitting produced nothing.
    return clipped or text


# Enforce the 400-character cap while preferring to keep original URLs.
def _cap_length(text: str, urls: list[str], max_chars: int = MAX_REWRITE_CHARS) -> str:
    """Return text truncated to max_chars, re-attaching URLs if they were cut."""

    # Fast path: already within the chat-scale cap.
    if len(text) <= max_chars:
        # Return the rewrite unchanged.
        return text
    # Build a suffix of URLs that must remain visible for honest link features.
    unique_urls: list[str] = []
    # Preserve first-seen URL order while dropping duplicates.
    for url in urls:
        # Skip URLs already queued for the suffix.
        if url not in unique_urls:
            # Queue this URL to be appended if missing after truncation.
            unique_urls.append(url)
    # Prefer a body budget that leaves room for every URL plus separators.
    url_suffix = (" " + " ".join(unique_urls)) if unique_urls else ""
    # Compute how many characters remain for prose if all URLs are forced in.
    budget = max_chars - len(url_suffix)
    # If URLs alone exceed the cap, keep as many full URLs as fit after a short tag.
    if budget < 12:
        # Start with a tiny chat wrapper so the row is not URL-only noise.
        prefix = "pls check "
        # Accumulate URLs that still fit in the cap.
        fitted: list[str] = []
        # Running length including the prefix and spaces.
        used = len(prefix)
        # Add URLs until the cap is reached.
        for url in unique_urls:
            # Count this URL plus a separating space if not the first.
            extra = len(url) + (1 if fitted else 0)
            # Stop when the next URL would exceed the cap.
            if used + extra > max_chars:
                # Do not include a partial URL.
                break
            # Accept this URL into the fitted list.
            fitted.append(url)
            # Account for the characters just consumed.
            used += extra
        # Return the prefix plus whatever URLs fit.
        return (prefix + " ".join(fitted)).strip()[:max_chars]
    # Truncate the prose body on a word boundary when possible.
    body = text[:budget].rstrip()
    # Prefer cutting at the last space so the last token is not a fragment.
    last_space = body.rfind(" ")
    # Only apply the word-boundary cut when it does not throw away most of the body.
    if last_space >= max(12, budget // 3):
        # Keep the complete last word.
        body = body[:last_space].rstrip()
    # Append URLs that are not already present in the truncated body.
    missing = [url for url in unique_urls if url not in body]
    # Join the truncated body with any missing URLs.
    combined = (body + (" " + " ".join(missing) if missing else "")).strip()
    # Hard-slice in case joining still overshot by a few characters.
    return combined[:max_chars]


# Re-attach any original URLs the cleaner dropped, using chat phrasing on scams.
def _ensure_urls(text: str, urls: list[str], label: int) -> str:
    """Return text that still contains every original URL when possible."""

    # Start from the current rewrite.
    output = text
    # Walk original URLs in first-seen order.
    for index, url in enumerate(urls):
        # Skip URLs that already appear in the rewritten body.
        if url in output:
            # This URL is already visible to TF-IDF and the URL-feature head.
            continue
        # Legitimate rows keep the raw URL; do not wrap with phishing verbs.
        if label == LEGITIMATE_LABEL:
            # Append the URL as a plain chat link share.
            output = (output + " " + url).strip()
            # Move on to the next missing URL.
            continue
        # Scam rows may wrap the first missing URL in chat phrasing.
        if index == 0:
            # Choose a wrapper from a tiny frozen list keyed by URL length (deterministic).
            wrappers = (
                f"can you open this {url}",
                f"pls check {url}",
                f"verify here: {url}",
            )
            # Pick a wrapper without hashing the eval-set wording.
            wrapper = wrappers[len(url) % len(wrappers)]
            # Append the wrapped URL as a second DM sentence.
            output = (output + " " + wrapper).strip()
        else:
            # Additional scam URLs are appended raw so none are dropped.
            output = (output + " " + url).strip()
    # Return the URL-preserving rewrite.
    return output


# Add a light scam-DM opener when the row is labeled scam and still looks formal.
def _maybe_scam_opener(text: str, label: int) -> str:
    """Prefix a casual opener on scam rows that do not already start like a DM."""

    # Legitimate rows must not gain scam-style openers or urgency.
    if label != SCAM_LABEL or not text:
        # Return the rewrite unchanged.
        return text
    # Detect messages that already start with chat-register openers.
    if re.match(r"^(?:hey|hi|yo|sup|omg|pls|please|yo,)\b", text, flags=re.IGNORECASE):
        # Keep the existing casual start.
        return text
    # Pick a deterministic opener from the frozen list (not eval-set wording).
    opener = _SCAM_OPENERS[len(text) % len(_SCAM_OPENERS)]
    # Lowercase the first character after the opener when it is a Latin letter.
    rest = text[0].lower() + text[1:] if text[0].isalpha() else text
    # Return the opener plus the existing body.
    return opener + rest


# Convert one message into a short DM-style line, or None if nothing usable remains.
def rewrite_message(text: str, label: int) -> str | None:
    """Return a chat-register rewrite that preserves label meaning and URLs.

    Empty results are returned as None so the CLI can drop them and log counts.
    The original binary label is not returned because callers must copy it
    unchanged from the source row.
    """

    # Treat missing text as unusable.
    if text is None:
        # Signal the CLI to drop this row.
        return None
    # Work on a stripped string copy.
    original = str(text).strip()
    # Drop empty source rows before any regex work.
    if not original:
        # Signal the CLI to drop this row.
        return None
    # Bound regex cost on pathological multi-megabyte bodies after URL harvest.
    scan_window = original if len(original) <= 50_000 else original[:50_000]
    # Harvest URLs (including hrefs) from the original so stripping cannot lose them.
    urls = extract_urls(scan_window)
    # Limit artifact stripping to a bounded window of the original body.
    work = original if len(original) <= 20_000 else original[:20_000]
    # Strip headers, quotes, unsubscribes, and disclaimers BEFORE shortening.
    cleaned = strip_email_artifacts(work)
    # If stripping removed everything, fall back to a short slice of the original.
    if not cleaned:
        # Use a small original window so the row is not dropped solely due to chrome.
        cleaned = _WHITESPACE_RE.sub(" ", work[:400]).strip()
    # Protect URLs so contractions cannot rewrite inside them.
    protected, mapping = _protect_urls(cleaned, urls)
    # Informalize the protected body (contractions, pls, thanks).
    informal = _informalize(protected)
    # Restore URLs before sentence clipping so they count as part of the DM.
    informal = _restore_urls(informal, mapping)
    # Keep 1–3 sentences of chat-scale text.
    shortened = _first_sentences(informal)
    # Re-attach any URLs the cleaner dropped, with scam-only chat wrapping.
    with_urls = _ensure_urls(shortened, urls, label)
    # Add a light scam-DM opener when appropriate; never on legitimate rows.
    with_opener = _maybe_scam_opener(with_urls, label)
    # Enforce the 400-character cap while preferring to keep real URLs.
    capped = _cap_length(with_opener, urls)
    # Collapse whitespace introduced by wrapping and truncation.
    final = _WHITESPACE_RE.sub(" ", capped).strip()
    # Drop empty results so they never enter processed_chat.
    if not final:
        # Signal the CLI to count this row as dropped.
        return None
    # Legitimate rewrites must not have grown phishing boilerplate that was absent.
    if label == LEGITIMATE_LABEL:
        # Compare against the original lowercased body for presence checks.
        original_lower = original.lower()
        # Inspect the rewrite in lowercase.
        final_lower = final.lower()
        # Reject accidental insertion of scam frames into ham (should not happen).
        for phrase in _HAM_FORBIDDEN_PHRASES:
            # Skip phrases that were already in the source (do not strip original meaning).
            if phrase in original_lower:
                # This phrase is original content, not rewriter-inserted.
                continue
            # If the rewriter introduced a forbidden phrase, strip that phrase only.
            if phrase in final_lower:
                # Remove the inserted phrase without touching URLs.
                final = re.sub(re.escape(phrase), "", final, flags=re.IGNORECASE)
                # Collapse leftover spaces after the removal.
                final = _WHITESPACE_RE.sub(" ", final).strip()
    # Return None if the safety strip emptied the message.
    return final or None
