"""Validate ciphertext-only WebSocket envelopes the server may persist and relay."""

# Import base64 to decode wire-format ciphertext and nonce without interpreting them.
import base64

# Import binascii's error type, which base64 raises on malformed padding/characters.
import binascii

# Import UUID so optional routing metadata can be type-checked against the path/auth user.
from uuid import UUID

# Import Pydantic's model base and field-level validation helpers.
from pydantic import BaseModel, Field, field_validator

# XChaCha20-Poly1305 IETF public nonce length (crypto_aead_xchacha20poly1305_ietf_NPUBBYTES).
XCHACHA20POLY1305_IETF_NPUBBYTES = 24
# XChaCha20-Poly1305 IETF authentication-tag length; ciphertext is at least this long.
XCHACHA20POLY1305_IETF_ABYTES = 16
# Cap stored envelopes so a single frame cannot fill the database with unbounded bytes.
MAX_CIPHERTEXT_BYTES = 32768


# Validate the ciphertext-only envelope a connected client may send over the WebSocket.
class RelayEnvelopeIn(BaseModel):
    """Represent one client-submitted ciphertext envelope.

    The server never decrypts these fields. Optional conversation_id and
    sender_id are accepted only so a mismatch with the authenticated
    WebSocket context can be rejected at the protocol layer (A1's
    cryptographic verify of associated data still happens client-side).
    """

    # Carry the AEAD ciphertext (including the Poly1305 tag) as standard base64.
    ciphertext: str = Field(min_length=1, max_length=65536)
    # Carry the public nonce as standard base64.
    nonce: str = Field(min_length=1, max_length=128)
    # Carry the non-secret epoch id the sender used to derive the message key.
    key_epoch: int = Field(ge=0, le=2_147_483_647)
    # Optional routing claim; if present it must match the WebSocket path's conversation.
    conversation_id: UUID | None = None
    # Optional sender claim; if present it must match the authenticated user.
    sender_id: UUID | None = None
    # Optional client-generated message identity (v2 associated data only).
    #
    # Present on every v2 send, new or edit. The client must choose this
    # *before* encrypting, since it is bound into the AEAD tag — the server
    # cannot assign an id after the fact the way it does the storage-only
    # `id` primary key. Omitted entirely, this frame is a v1 send (unchanged
    # from before message editing existed) and is never treated as an edit.
    message_id: UUID | None = None
    # Count edits to one v2 message: 0 for the original send. A resend whose
    # `message_id` matches an existing row in this conversation is only
    # accepted as an edit when `revision` is exactly that row's stored
    # revision + 1 (see app/services/relay.py) — this is what closes the
    # edit-rollback gap: the server cannot be tricked into "editing" out of
    # order, and a stale ciphertext can never be replayed as the current one.
    revision: int = Field(default=0, ge=0, le=2_147_483_647)

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
        if len(decoded) > MAX_CIPHERTEXT_BYTES:
            # Bound stored envelope size independently of whatever the client claimed.
            raise ValueError("ciphertext exceeds the maximum envelope size")
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


# Describe the ciphertext-only envelope the server fans out to the peer.
class RelayEnvelopeOut(BaseModel):
    """Represent one persisted envelope as relayed to conversation members.

    Contains only ciphertext, nonce, key_epoch, and routing metadata.
    Never includes a message body or any key material.
    """

    # Mark this frame as a stored envelope so clients can distinguish errors/acks.
    type: str = "envelope"
    # Identify the persisted row so the recipient can key the UI list.
    id: UUID
    # Identify the conversation this envelope belongs to (AEAD associated data).
    conversation_id: UUID
    # Identify the sender this envelope claims (AEAD associated data).
    sender_id: UUID
    # Carry the AEAD ciphertext as standard base64.
    ciphertext: str
    # Carry the public nonce as standard base64.
    nonce: str
    # Carry the non-secret epoch id the sender used.
    key_epoch: int
    # Carry the insertion timestamp as an ISO-8601 string for display ordering.
    created_at: str
    # Carry the AD format version so the recipient reconstructs the right associated
    # data before calling decrypt (v1 = original 4-field AD; v2 = adds message
    # identity + revision). Defaults to 1 so a client that predates editing still
    # gets a well-formed frame if it somehow receives one (it will simply never see
    # a v2 row, since only an updated client sends message_id).
    ad_version: int = 1
    # Carry the client-chosen message identity for v2 rows; null for v1 rows.
    message_id: UUID | None = None
    # Carry the current revision for v2 rows; 0 for v1 rows (which have no concept
    # of a revision at all).
    revision: int = 0
    # Carry the most recent edit time as an ISO-8601 string, or null if never edited.
    edited_at: str | None = None
    # True when the peer's device has acknowledged this envelope (sender-view only).
    peer_delivered: bool = False
    # True when the peer has focused the chat on this envelope (sender-view only).
    peer_read: bool = False


# Describe the broadcast frame sent to a conversation's members after a hard
# delete-for-everyone. Carries only the deleted row's id — never its ciphertext,
# which the sender already discarded from local state before calling delete.
class MessageDeletedOut(BaseModel):
    """Represent one hard-deleted envelope's id, broadcast to the peer."""

    # Mark this frame as a deletion so clients can distinguish it from an envelope.
    type: str = "message_deleted"
    # Identify which conversation this deletion applies to (routing only).
    conversation_id: UUID
    # Identify the now-removed row so the recipient can drop it from their transcript.
    id: UUID


# Wrap conversation-scoped history so the client never queries the whole messages table.
class MessageHistoryResponse(BaseModel):
    """Represent one conversation's stored envelopes in chronological order.

    Slice 7 serves GET /conversations/{id}/messages from this shape. Every
    item is ciphertext-only; classification scores never appear here.
    """

    # Carry envelopes oldest-first so the chat transcript can render top-to-bottom.
    messages: list[RelayEnvelopeOut]
    # Opaque id of the last envelope in this page when more history remains; null
    # means this response is complete. Omitted pagination (no limit) always null.
    next_cursor: UUID | None = None
