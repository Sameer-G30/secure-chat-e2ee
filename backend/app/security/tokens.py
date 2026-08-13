"""Issue and verify PyJWT-signed access and refresh tokens (decision A4).

Access tokens are short-lived and carry no server-side state. Refresh
tokens are also JWTs (so their signature and expiry are self-verifying),
but the server additionally stores a SHA-256 hash of each issued refresh
token in the `refresh_tokens` table so it can enforce single-use rotation
and detect replay of an already-rotated token.
"""

# Import hashlib to compute the stored digest of an issued refresh token.
import hashlib

# Import uuid to type user identifiers and mint unique refresh-token ids.
import uuid

# Import dataclass to bundle a freshly minted refresh token's fields together.
from dataclasses import dataclass

# Import timezone-aware datetime helpers; naive datetimes must never leak into JWT claims.
from datetime import UTC, datetime, timedelta

# Import PyJWT itself (decision A4: python-jose is unmaintained with known CVEs).
import jwt

# Import validated settings so secrets and expiries never come from a hardcoded value.
from app.config import get_settings

# Name the two token kinds so a stolen access token cannot be replayed as a refresh token.
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


# Distinguish "this token is unusable" from every other exception type.
class TokenError(Exception):
    """Raise when a token fails signature verification, has expired, or has the wrong claims."""


# Bundle everything the login/refresh endpoints need after minting a refresh token.
@dataclass(frozen=True)
class IssuedRefreshToken:
    """Represent one freshly minted refresh token and its persisted metadata."""

    # Carry the raw JWT string returned to the client; never persisted verbatim.
    raw_token: str
    # Carry the SHA-256 hex digest that is safe to store and compare against.
    token_hash: str
    # Carry the unique token id embedded in the JWT, useful for audit logging.
    jti: str
    # Carry the expiry so the caller can persist it alongside the hash.
    expires_at: datetime


# Read the current time once per call, always timezone-aware.
def _utc_now() -> datetime:
    """Return the current UTC time as an aware datetime."""

    return datetime.now(UTC)


# Mint a short-lived access token proving the bearer authenticated as this user.
def create_access_token(user_id: uuid.UUID, username: str) -> str:
    """Return a signed JWT access token valid for the configured short lifetime."""

    # Read the signing secret, algorithm, and lifetime from validated settings.
    settings = get_settings()
    issued_at = _utc_now()
    payload = {
        # Identify the account this token authenticates.
        "sub": str(user_id),
        # Carry the username so the API rarely needs an extra database round trip.
        "username": username,
        # Mark this token's kind so it cannot be reused where a refresh token is expected.
        "type": ACCESS_TOKEN_TYPE,
        # Record issuance time for auditability.
        "iat": issued_at,
        # Expire quickly to limit the damage window of a leaked access token.
        "exp": issued_at + timedelta(minutes=settings.access_token_expire_minutes),
    }
    # Sign with the configured secret and algorithm; never accept "alg": "none".
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


# Mint a longer-lived refresh token and the metadata needed to persist its hash.
def create_refresh_token(user_id: uuid.UUID) -> IssuedRefreshToken:
    """Return a signed refresh JWT plus the hash/expiry the caller must persist."""

    # Read the signing secret, algorithm, and lifetime from validated settings.
    settings = get_settings()
    issued_at = _utc_now()
    # Generate a unique token id so rotation and revocation can target one exact token.
    jti = str(uuid.uuid4())
    expires_at = issued_at + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        # Identify the account this refresh token belongs to.
        "sub": str(user_id),
        # Mark this token's kind so an access token can never be replayed as a refresh token.
        "type": REFRESH_TOKEN_TYPE,
        # Carry a unique id so the stored hash maps back to exactly one issuance.
        "jti": jti,
        # Record issuance time for auditability.
        "iat": issued_at,
        # Expire after the configured longer lifetime.
        "exp": expires_at,
    }
    raw_token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return IssuedRefreshToken(
        raw_token=raw_token,
        token_hash=hash_refresh_token(raw_token),
        jti=jti,
        expires_at=expires_at,
    )


# Compute the digest stored in the database in place of the raw refresh token.
def hash_refresh_token(raw_token: str) -> str:
    """Return the SHA-256 hex digest of a raw refresh token string.

    Storing only this hash means a database read (backup, dump, injection)
    cannot be turned directly into a usable refresh token, mirroring how
    passwords are never stored in recoverable form.
    """

    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


# Verify a token's signature, expiry, and declared type before trusting its claims.
def decode_token(raw_token: str, *, expected_type: str) -> dict[str, object]:
    """Return decoded claims, raising TokenError on any invalid or mistyped token."""

    settings = get_settings()
    try:
        # Verify the signature and standard exp/iat claims in one call.
        payload = jwt.decode(
            raw_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.PyJWTError as exc:
        # Collapse every PyJWT failure mode into one caller-facing error type.
        raise TokenError(
            "token is missing, malformed, expired, or has an invalid signature"
        ) from exc
    # Reject a syntactically valid token minted for the wrong purpose.
    if payload.get("type") != expected_type:
        raise TokenError(f"expected a '{expected_type}' token but received a different type")
    return payload
