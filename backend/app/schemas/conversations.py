"""Validate conversation create/fetch payloads and shape ciphertext-free responses."""

# Import datetime for the response's typed timestamp field.
from datetime import datetime

# Import UUID for typed conversation and account identifiers.
from uuid import UUID

# Import Pydantic's model base and field-level validation helpers.
from pydantic import BaseModel, Field

# Reuse the same username bounds the auth schema already documents.
_USERNAME_MIN_LENGTH = 3
_USERNAME_MAX_LENGTH = 32


# Validate the single field a conversation start-or-fetch request must supply.
class CreateConversationRequest(BaseModel):
    """Represent the client-submitted 1:1 conversation lookup payload."""

    # Identify the other participant by the same handle used for GET /keys/{username}.
    peer_username: str = Field(min_length=_USERNAME_MIN_LENGTH, max_length=_USERNAME_MAX_LENGTH)


# Describe one participant as returned to an authenticated conversation member.
class ConversationParticipant(BaseModel):
    """Represent one side of a 1:1 conversation without any private key material."""

    # Identify the account with the same UUID used as AEAD associated data.
    id: UUID
    # Identify the account with the handle the UI displays.
    username: str
    # Return the peer's public key when present; the server never has a private key.
    public_key: str | None = None


# Describe a 1:1 conversation the caller is a member of.
class ConversationResponse(BaseModel):
    """Represent one conversation plus both participants' non-secret identity fields."""

    # Identify the conversation; clients bind this UUID into AEAD associated data.
    id: UUID
    # Return the non-secret epoch counter used for per-message KDF subkey ids.
    current_epoch: int
    # Record when the conversation row was first created.
    created_at: datetime
    # Identify the authenticated caller so the client does not have to decode its JWT.
    self: ConversationParticipant
    # Identify the other participant the caller asked to chat with.
    peer: ConversationParticipant

    # Allow constructing this schema from mixed ORM/computed fields.
    model_config = {"from_attributes": True}


# Describe the non-secret epoch counter returned by the epoch endpoint.
class EpochResponse(BaseModel):
    """Represent the server-coordinated epoch integer for one conversation.

    This value is not secret and is never a key. Clients pass it to
    crypto_kdf_derive_from_key; the server never derives the key itself.
    """

    # Identify which conversation this counter belongs to.
    conversation_id: UUID
    # Return the current non-secret epoch integer (starts at 0; Slice 8 bumps it).
    current_epoch: int
