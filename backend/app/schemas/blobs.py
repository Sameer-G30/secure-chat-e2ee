"""Validate opaque encrypted-blob upload/download payloads the server never opens."""

# Import base64 to decode wire-format ciphertext and nonce without interpreting them.
import base64

# Import binascii's error type, which base64 raises on malformed padding/characters.
import binascii

# Import datetime for the download response timestamp.
from datetime import datetime

# Import UUID for the client-chosen blob identity.
from uuid import UUID

# Import Pydantic's model base and field-level validation helpers.
from pydantic import BaseModel, Field, field_validator

# Reuse the public nonce length already enforced on chat envelopes.
from app.schemas.messages import XCHACHA20POLY1305_IETF_ABYTES, XCHACHA20POLY1305_IETF_NPUBBYTES

# Cap stored file ciphertext so one upload cannot fill the database.
MAX_BLOB_CIPHERTEXT_BYTES = 1_500_000


# Validate the client-encrypted image bytes a member may store for one conversation.
class EncryptedBlobIn(BaseModel):
    """Represent one client-submitted sealed file.

    id is chosen before encryption so associated data can bind it. The server
    never decrypts ciphertext or nonce.
    """

    # Carry the client-chosen blob identity bound into file associated data.
    id: UUID
    # Carry the AEAD ciphertext (including the Poly1305 tag) as standard base64.
    ciphertext: str = Field(min_length=1, max_length=2_097_152)
    # Carry the public nonce as standard base64.
    nonce: str = Field(min_length=1, max_length=128)

    # Reject ciphertext that is not valid standard base64 of a plausible AEAD payload.
    @field_validator("ciphertext")
    @classmethod
    def ciphertext_must_be_plausible_aead_bytes(cls, value: str) -> str:
        """Require standard base64 that decodes to a tag-sized-or-larger blob."""

        try:
            # Require strict standard base64 (matches libsodium's ORIGINAL variant).
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            # Reject malformed base64 before it ever reaches storage.
            raise ValueError("ciphertext must be valid base64") from exc
        if len(decoded) < XCHACHA20POLY1305_IETF_ABYTES:
            # A real XChaCha20-Poly1305 ciphertext is at least the 16-byte tag.
            raise ValueError("ciphertext is too short to be an authenticated envelope")
        if len(decoded) > MAX_BLOB_CIPHERTEXT_BYTES:
            # Bound stored file size independently of whatever the client claimed.
            raise ValueError("ciphertext exceeds the maximum blob size")
        # Return the original base64 text; persistence decodes it once, after authz checks.
        return value

    # Reject a nonce that cannot be the 24-byte XChaCha20-Poly1305 IETF nonce.
    @field_validator("nonce")
    @classmethod
    def nonce_must_be_xchacha20_ietf_length(cls, value: str) -> str:
        """Require standard base64 that decodes to exactly 24 bytes."""

        try:
            # Require strict standard base64 (matches libsodium's ORIGINAL variant).
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            # Reject malformed base64 before it ever reaches storage.
            raise ValueError("nonce must be valid base64") from exc
        if len(decoded) != XCHACHA20POLY1305_IETF_NPUBBYTES:
            # Reject any length that could not be a real XChaCha20-Poly1305 IETF nonce.
            raise ValueError(
                f"nonce must decode to exactly {XCHACHA20POLY1305_IETF_NPUBBYTES} bytes"
            )
        # Return the original base64 text; persistence decodes it once, after authz checks.
        return value


# Describe the opaque blob the server returns to a conversation member.
class EncryptedBlobOut(BaseModel):
    """Represent one stored sealed file without opening it."""

    # Identify the blob with the client-chosen UUID.
    id: UUID
    # Identify which conversation this blob belongs to.
    conversation_id: UUID
    # Identify which member uploaded the opaque bytes.
    uploader_id: UUID
    # Carry the AEAD ciphertext as standard base64.
    ciphertext: str
    # Carry the public nonce as standard base64.
    nonce: str
    # Carry the stored byte length so the client can sanity-check before decrypt.
    byte_length: int
    # Carry the insertion timestamp as an ISO-8601 datetime.
    created_at: datetime
