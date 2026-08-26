"""Shared username bounds used by register, contacts, conversations, and key lookup."""

# Import Annotated so Pydantic can attach validators to a reusable username type.
from typing import Annotated

# Import AfterValidator and Field so length and character rules live in one place.
from pydantic import AfterValidator, Field

# Bound the username so it is usable in URLs (public key lookup) and readable in UI.
USERNAME_MIN_LENGTH = 3
# Cap the username so oversized handles cannot bloat indexes or path segments.
USERNAME_MAX_LENGTH = 32
# Restrict handles to characters that need no percent-encoding in a URL path.
USERNAME_PATTERN = r"^[A-Za-z0-9_-]+$"


# Reject empty, whitespace, or punctuation-heavy handles before they reach storage.
def require_url_safe_username(value: str) -> str:
    """Keep usernames safe to embed in /keys/{username} and contact handles."""

    # Restrict to letters, digits, underscore, and hyphen only.
    if not all(character.isalnum() or character in "_-" for character in value):
        # Reject anything else before it ever reaches the database.
        raise ValueError("username may only contain letters, digits, '_' and '-'")
    # Return the validated value unchanged.
    return value


# Reuse this type on register, add-contact, and start-conversation request bodies.
Username = Annotated[
    # Store the handle as plain text after the length and character checks pass.
    str,
    # Enforce the documented length floor and cap at the schema layer.
    Field(min_length=USERNAME_MIN_LENGTH, max_length=USERNAME_MAX_LENGTH),
    # Apply the URL-safe character rule after length validation.
    AfterValidator(require_url_safe_username),
]
