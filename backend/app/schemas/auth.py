"""Validate registration input and shape the account data returned to clients."""

# Import datetime for the response's typed timestamp field.
from datetime import datetime

# Import UUID for the response's typed identifier field.
from uuid import UUID

# Import Pydantic's model base, email type, and field-level validation helpers.
from pydantic import BaseModel, EmailStr, Field, field_validator

# Import the shared minimum-length policy so the API and hasher stay in sync.
from app.security.passwords import MINIMUM_PASSWORD_LENGTH

# Bound the username so it is usable in URLs (public key lookup) and readable in UI.
_USERNAME_MIN_LENGTH = 3
_USERNAME_MAX_LENGTH = 32


# Validate the exact fields a new account must supply.
class RegisterRequest(BaseModel):
    """Represent the client-submitted registration payload."""

    # Require a handle usable later for GET /keys/{username} lookups.
    username: str = Field(min_length=_USERNAME_MIN_LENGTH, max_length=_USERNAME_MAX_LENGTH)
    # Require a syntactically valid, normalizable email address.
    email: EmailStr
    # Require a minimum-length password; Argon2id, not complexity rules, does the real work.
    password: str = Field(min_length=MINIMUM_PASSWORD_LENGTH, max_length=256)

    # Reject usernames that are not plain alphanumeric/underscore/hyphen handles.
    @field_validator("username")
    @classmethod
    def username_must_be_url_safe(cls, value: str) -> str:
        """Keep usernames safe to embed in a future /keys/{username} path segment."""

        # Restrict to characters that need no percent-encoding in a URL path.
        if not all(character.isalnum() or character in "_-" for character in value):
            # Reject anything else before it ever reaches the database.
            raise ValueError("username may only contain letters, digits, '_' and '-'")
        # Return the validated value unchanged.
        return value


# Describe the account fields safe to return after registration.
class RegisterResponse(BaseModel):
    """Represent the account data returned to a newly registered client."""

    # Identify the created account for subsequent authenticated requests.
    id: UUID
    # Echo back the stored username for immediate UI confirmation.
    username: str
    # Echo back the stored email for immediate UI confirmation.
    email: str
    # Report creation time; never include password_hash or public_key secrets here.
    created_at: datetime

    # Allow constructing this schema directly from the SQLAlchemy ORM instance.
    model_config = {"from_attributes": True}
