"""Validate the base64 X25519 public-key payloads accepted and returned by /keys."""

# Import base64 to validate that an uploaded key decodes to the expected byte length.
import base64

# Import binascii's error type, which base64 raises on malformed padding/characters.
import binascii

# Import Pydantic's model base and field-level validation helpers.
from pydantic import BaseModel, Field, field_validator

# An X25519 public key is always exactly 32 raw bytes.
X25519_PUBLIC_KEY_BYTES = 32


# Validate the single field a key-upload request must supply.
class PublicKeyUploadRequest(BaseModel):
    """Represent the client-submitted public-key upload payload.

    The server only ever sees this public key — the matching private key
    is generated and sealed entirely client-side and is never transmitted.
    """

    # Accept the client's base64-encoded X25519 public key.
    public_key: str = Field(min_length=1, max_length=512)

    # Reject anything that cannot possibly be a real X25519 public key.
    @field_validator("public_key")
    @classmethod
    def must_decode_to_x25519_key_length(cls, value: str) -> str:
        """Require standard base64 that decodes to exactly 32 bytes."""

        try:
            # Require strict standard base64 (matches libsodium's ORIGINAL variant).
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            # Reject malformed base64 before it ever reaches storage.
            raise ValueError("public_key must be valid base64") from exc
        if len(decoded) != X25519_PUBLIC_KEY_BYTES:
            # Reject any length that could not be a real X25519 public key.
            raise ValueError(f"public_key must decode to exactly {X25519_PUBLIC_KEY_BYTES} bytes")
        # Return the original base64 text unchanged; storage keeps the client's encoding.
        return value


# Describe the public-key fields returned by both upload and lookup.
class PublicKeyResponse(BaseModel):
    """Represent one account's public identity as returned to any authenticated caller."""

    # Identify whose public key this is.
    username: str
    # Return the base64 X25519 public key; this value is not secret.
    public_key: str

    # Allow constructing this schema directly from the SQLAlchemy ORM instance.
    model_config = {"from_attributes": True}
