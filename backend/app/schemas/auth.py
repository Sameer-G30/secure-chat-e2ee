"""Validate registration input and shape the account data returned to clients."""

# Import datetime for the response's typed timestamp field.
from datetime import datetime

# Import UUID for the response's typed identifier field.
from uuid import UUID

# Import Pydantic's model base, email type, and field-level validation helpers.
from pydantic import BaseModel, EmailStr, Field

# Import the shared username type so register uses the same handle rules as contacts.
from app.schemas.usernames import USERNAME_MAX_LENGTH, Username

# Import the shared minimum-length policy so the API and hasher stay in sync.
from app.security.passwords import MINIMUM_PASSWORD_LENGTH

# RFC 5321 practical cap so a registration body cannot carry an unbounded mailbox string.
_EMAIL_MAX_LENGTH = 254
# Cap refresh JWTs so a huge body cannot be used as an unauthenticated dump target.
REFRESH_TOKEN_MAX_LENGTH = 8192


# Validate the exact fields a new account must supply.
class RegisterRequest(BaseModel):
    """Represent the client-submitted registration payload."""

    # Require a handle usable later for GET /keys/{username} lookups.
    username: Username
    # Require a syntactically valid, length-capped, normalizable email address.
    email: EmailStr = Field(max_length=_EMAIL_MAX_LENGTH)
    # Require a minimum-length password; Argon2id, not complexity rules, does the real work.
    password: str = Field(min_length=MINIMUM_PASSWORD_LENGTH, max_length=256)


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


# Validate the exact fields a login attempt must supply.
class LoginRequest(BaseModel):
    """Represent the client-submitted login payload."""

    # Identify which account to authenticate.
    username: str = Field(min_length=1, max_length=USERNAME_MAX_LENGTH)
    # Accept the plaintext password only for the duration of this HTTPS request.
    #
    # Deliberately no min_length policy re-check here: enforcing the registration
    # password policy on login would leak information about which usernames are
    # registered accounts versus not, for no security benefit (Argon2id
    # verification against a wrong-length password already just fails).
    password: str = Field(min_length=1, max_length=256)


# Validate the single field a token-refresh request must supply.
class RefreshRequest(BaseModel):
    """Represent the client-submitted refresh-rotation payload."""

    # Carry the refresh token whose hash the server looks up and rotates.
    refresh_token: str = Field(min_length=1, max_length=REFRESH_TOKEN_MAX_LENGTH)


# Describe the token pair issued after a successful login or rotation.
class TokenPairResponse(BaseModel):
    """Represent one freshly issued access/refresh token pair."""

    # Carry the short-lived JWT used to authenticate subsequent API requests.
    access_token: str
    # Carry the longer-lived, single-use JWT used only to request a new pair.
    refresh_token: str
    # Name the standard bearer scheme so clients know how to send the access token.
    token_type: str = "bearer"
    # Tell the client whether this account still needs to complete key upload,
    # so it knows whether to run the client-side keypair-generation flow now.
    has_public_key: bool
