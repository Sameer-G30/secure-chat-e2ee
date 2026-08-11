"""Extract plain-text bodies from raw email bytes for corpus normalization."""

# Import email.message_from_bytes to parse RFC822 messages from archive members.
from email import message_from_bytes

# Import Message for type-checked multipart traversal helpers.
from email.message import Message

# Import HTMLParser to strip tags from text/html parts without adding dependencies.
from html.parser import HTMLParser


# Collect visible text while discarding markup and script-like noise.
class _VisibleTextExtractor(HTMLParser):
    """Accumulate readable characters from an HTML email body."""

    # Prepare an empty buffer before any HTML tokens arrive.
    def __init__(self) -> None:
        # Initialize the standard HTML parser base class.
        super().__init__()
        # Store concatenated visible text chunks.
        self.chunks: list[str] = []
        # Track whether the parser is currently inside a skipped element.
        self._skip_depth = 0

    # Ignore text nested inside script or style elements.
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Increase the skip depth when entering non-visible containers.
        if tag in {"script", "style"}:
            # Enter one more nested skipped region.
            self._skip_depth += 1

    # Resume collecting text after leaving skipped elements.
    def handle_endtag(self, tag: str) -> None:
        # Decrease the skip depth when leaving non-visible containers.
        if tag in {"script", "style"} and self._skip_depth:
            # Leave one nested skipped region.
            self._skip_depth -= 1

    # Append ordinary character data outside skipped regions.
    def handle_data(self, data: str) -> None:
        # Ignore character data that belongs to script or style blocks.
        if self._skip_depth:
            # Skip invisible markup content entirely.
            return
        # Retain visible HTML character data for classification features.
        self.chunks.append(data)


# Convert HTML markup into approximate plain text for model features.
def html_to_text(html: str) -> str:
    """Return whitespace-normalized text extracted from an HTML fragment."""

    # Construct a fresh extractor for this HTML payload.
    extractor = _VisibleTextExtractor()
    # Parse the HTML string while collecting visible text.
    extractor.feed(html)
    # Close any unfinished parser state before joining chunks.
    extractor.close()
    # Collapse runs of whitespace so lengths remain comparable across sources.
    return " ".join("".join(extractor.chunks).split())


# Decode a MIME part using its declared charset with a safe fallback.
def _decode_part_payload(part: Message) -> str:
    """Return a Unicode string for one email part payload."""

    # Prefer the decoded bytes interface over the legacy get_payload string path.
    payload = part.get_payload(decode=True)
    # Handle parts that only expose a Unicode string payload.
    if payload is None:
        # Read the undecoded payload for already-decoded text parts.
        raw_payload = part.get_payload()
        # Preserve string payloads directly.
        if isinstance(raw_payload, str):
            # Return the already-decoded text content.
            return raw_payload
        # Treat unexpected nested structures as empty rather than crashing.
        return ""
    # Read the charset declared by the MIME part, defaulting to UTF-8.
    charset = part.get_content_charset() or "utf-8"
    # Attempt the declared charset while tolerating illegal spam charset names.
    try:
        # Decode bytes with replacement so malformed spam still yields text.
        return payload.decode(charset, errors="replace")
    except LookupError:
        # Fall back to UTF-8 when the declared charset is unknown to Python.
        return payload.decode("utf-8", errors="replace")


# Prefer text/plain, then stripped text/html, then any textual fallback.
def extract_message_text(message: Message) -> str:
    """Return a single plain-text body from a parsed email.message.Message."""

    # Accumulate plain-text MIME parts in encounter order.
    plain_parts: list[str] = []
    # Accumulate HTML MIME parts when plain text is unavailable.
    html_parts: list[str] = []
    # Walk every MIME part including nested multiparts.
    for part in message.walk():
        # Skip container parts that only hold child MIME sections.
        if part.is_multipart():
            # Continue searching child parts for leaf content.
            continue
        # Read the MIME content type for routing plain versus HTML extraction.
        content_type = part.get_content_type()
        # Ignore attachments that are not textual message bodies.
        content_disposition = str(part.get("Content-Disposition", "")).lower()
        # Skip explicit file attachments while keeping inline bodies.
        if "attachment" in content_disposition:
            # Do not treat attached files as the message body.
            continue
        # Decode the leaf payload into Unicode text.
        decoded = _decode_part_payload(part)
        # Collect text/plain bodies as the preferred representation.
        if content_type == "text/plain":
            # Store the decoded plain-text body fragment.
            plain_parts.append(decoded)
        # Collect HTML bodies as a secondary representation.
        elif content_type == "text/html":
            # Store the decoded HTML body fragment for later stripping.
            html_parts.append(decoded)
    # Prefer concatenated plain-text parts when any exist.
    if plain_parts:
        # Normalize whitespace across joined plain-text fragments.
        return " ".join(" ".join(plain_parts).split())
    # Fall back to stripped HTML when the message has no text/plain part.
    if html_parts:
        # Convert HTML markup into approximate plain text.
        return html_to_text("\n".join(html_parts))
    # As a last resort, use the top-level subject so empty bodies remain auditable.
    subject = message.get("Subject", "")
    # Normalize the subject string when it is the only available text.
    return " ".join(str(subject).split())


# Prefer text/plain, then stripped text/html, then any textual fallback.
def extract_email_text(raw_message: bytes) -> str:
    """Return a single plain-text body suitable for scam-classification features."""

    # Parse the raw RFC822 bytes into a structured email message.
    message = message_from_bytes(raw_message)
    # Reuse the Message-based extractor for a single code path.
    return extract_message_text(message)
