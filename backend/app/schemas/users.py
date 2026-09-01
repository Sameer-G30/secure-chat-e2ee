"""Validate the authenticated username-search response.

Deliberately narrow: this replaces the legacy app's "download every user row
and substring-match in the browser" approach, so the response carries only
what a contact-add flow needs (a handle to add), never email, password hash,
or public key.
"""

# Import Pydantic's model base used by the response payload.
from pydantic import BaseModel

# Bound how many rows GET /users/search may return in one call.
MAX_USER_SEARCH_RESULTS = 20
# Require at least this many characters before searching (matches the legacy
# React app's `useContacts.js` minimum, which this endpoint replaces).
MIN_USER_SEARCH_QUERY_LENGTH = 2


# Describe one matched account. Username only — no email, key, or hash.
class UserSearchResult(BaseModel):
    """Represent one account found by a case-insensitive username prefix search."""

    # Carry only the handle a contact-add flow needs.
    username: str


# Describe the full set of matches for one search query.
class UserSearchResponse(BaseModel):
    """Represent every account whose username starts with the search query."""

    # Carry matched usernames, capped at MAX_USER_SEARCH_RESULTS.
    users: list[UserSearchResult]
