"""Persist and fan out ciphertext envelopes without ever decrypting them."""

# Import base64 to decode inbound wire format and re-encode outbound frames.
import base64

# Import UUID for typed conversation and sender identifiers.
from uuid import UUID

# Import FastAPI's WebSocket type so the hub can address connected peers.
from fastapi import WebSocket

# Import SQLAlchemy query helpers used for conversation-scoped reads.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM models this service reads and writes.
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User

# Import the validated inbound envelope and the outbound relay frame.
from app.schemas.messages import RelayEnvelopeIn, RelayEnvelopeOut

# Import conversation membership and public-key enforcement helpers.
from app.services.conversations import (
    get_conversation_for_participant,
    other_user_id,
    require_both_public_keys,
)


# Raised when an envelope's routing metadata does not match the authenticated context.
class EnvelopeRejected(Exception):
    """Reject a frame at the protocol layer without attempting any cryptographic verify."""


# Hold in-process WebSocket connections grouped by conversation, then by user.
class ConnectionHub:
    """Fan ciphertext frames to connected members of one conversation.

    This is routing only: the hub never inspects ciphertext bytes as plaintext
    and never holds a decryption key.
    """

    def __init__(self) -> None:
        # Map conversation_id -> user_id -> live sockets for that user in that room.
        self._rooms: dict[UUID, dict[UUID, set[WebSocket]]] = {}

    def reset(self) -> None:
        """Drop every tracked socket; used by tests so rooms cannot leak across cases."""

        # Clear the nested mapping rather than iterating sockets (tests close them).
        self._rooms.clear()

    def join(self, conversation_id: UUID, user_id: UUID, websocket: WebSocket) -> None:
        """Register a live socket as a member of one conversation."""

        # Create the conversation room on first join rather than pre-allocating.
        room = self._rooms.setdefault(conversation_id, {})
        # Create this user's socket set on first join from that account.
        room.setdefault(user_id, set()).add(websocket)

    def leave(self, conversation_id: UUID, user_id: UUID, websocket: WebSocket) -> None:
        """Forget a socket when it disconnects, dropping empty rooms as we go."""

        room = self._rooms.get(conversation_id)
        if room is None:
            return
        sockets = room.get(user_id)
        if sockets is None:
            return
        sockets.discard(websocket)
        if not sockets:
            room.pop(user_id, None)
        if not room:
            self._rooms.pop(conversation_id, None)

    async def broadcast(
        self,
        conversation_id: UUID,
        payload: dict[str, object],
        *,
        exclude_user_id: UUID | None = None,
    ) -> None:
        """Send a JSON frame to every connected member except an optional excluded user."""

        room = self._rooms.get(conversation_id, {})
        # Snapshot items so a disconnect during send cannot mutate the iterator.
        for user_id, sockets in list(room.items()):
            if exclude_user_id is not None and user_id == exclude_user_id:
                # The sender already has plaintext locally; do not echo ciphertext back.
                continue
            for websocket in list(sockets):
                await websocket.send_json(payload)


# Process-wide hub used by the WebSocket router; tests call reset() between cases.
connection_hub = ConnectionHub()


# Decode a validated envelope's base64 fields into raw bytes for BYTEA storage.
def decode_envelope_bytes(envelope: RelayEnvelopeIn) -> tuple[bytes, bytes]:
    """Return (ciphertext_bytes, nonce_bytes) from already-validated base64 text."""

    # Validators already checked length and alphabet; decode is then infallible.
    ciphertext = base64.b64decode(envelope.ciphertext, validate=True)
    nonce = base64.b64decode(envelope.nonce, validate=True)
    return ciphertext, nonce


# Reject routing claims that do not match the authenticated WebSocket context.
def assert_envelope_routing(
    envelope: RelayEnvelopeIn,
    conversation_id: UUID,
    sender_id: UUID,
    current_epoch: int,
) -> None:
    """Reject cross-conversation or spoofed-sender frames before persistence.

    Cryptographic authentication of associated data remains client-side (A1).
    This check is the server's protocol-layer equivalent: it will not store or
    fan out an envelope that claims the wrong conversation or sender.
    """

    if envelope.conversation_id is not None and envelope.conversation_id != conversation_id:
        # A frame routed on conversation A must not claim to belong to conversation B.
        raise EnvelopeRejected("envelope conversation_id does not match this connection")
    if envelope.sender_id is not None and envelope.sender_id != sender_id:
        # A caller cannot claim to send as the peer; sender_id is taken from the access token.
        raise EnvelopeRejected("envelope sender_id does not match the authenticated user")
    if envelope.key_epoch > current_epoch:
        # Clients may use the current or a past epoch, never one the server has not reached.
        raise EnvelopeRejected("key_epoch is ahead of the conversation's current epoch")


# Persist one ciphertext envelope scoped to a single conversation_id.
async def persist_envelope(
    db: AsyncSession,
    *,
    conversation_id: UUID,
    sender_id: UUID,
    ciphertext: bytes,
    nonce: bytes,
    key_epoch: int,
) -> Message:
    """Insert one envelope row. Callers must already have authorized membership.

    The INSERT always includes conversation_id so a later SELECT scoped by
    conversation_id can never return another conversation's rows.
    """

    message = Message(
        conversation_id=conversation_id,
        sender_id=sender_id,
        ciphertext=ciphertext,
        nonce=nonce,
        key_epoch=key_epoch,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


# Read envelopes for exactly one conversation; never the whole messages table.
async def list_envelopes_for_conversation(
    db: AsyncSession,
    conversation_id: UUID,
) -> list[Message]:
    """Return stored envelopes for one conversation_id, newest first.

    This helper exists so every messages-table read is conversation-scoped
    (§2, §11). Slice 4's UI does not paginate history; tests use this to
    prove a query for conversation A cannot see conversation B.
    """

    result = await db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
    )
    return list(result.all())


# Serialize a stored envelope for the WebSocket without adding plaintext fields.
def serialize_envelope(message: Message) -> RelayEnvelopeOut:
    """Return the ciphertext-only frame the peer will decrypt locally."""

    created_at = message.created_at.isoformat() if message.created_at is not None else ""
    return RelayEnvelopeOut(
        id=message.id,
        conversation_id=message.conversation_id,
        sender_id=message.sender_id,
        ciphertext=base64.b64encode(message.ciphertext).decode("ascii"),
        nonce=base64.b64encode(message.nonce).decode("ascii"),
        key_epoch=message.key_epoch,
        created_at=created_at,
    )


# Authorize, persist, and fan out one inbound WebSocket envelope.
async def relay_envelope(
    db: AsyncSession,
    *,
    conversation: Conversation,
    sender: User,
    envelope: RelayEnvelopeIn,
) -> RelayEnvelopeOut:
    """Validate routing, store ciphertext only, and return the outbound frame.

    The server never decrypts. Associated-data verification is the client's job.
    """

    # Reject claimed conversation/sender/epoch that do not match this connection.
    assert_envelope_routing(envelope, conversation.id, sender.id, conversation.current_epoch)
    # Decode wire base64 into BYTEA columns; lengths were already validated.
    ciphertext, nonce = decode_envelope_bytes(envelope)
    # Insert scoped by this conversation_id; never a global messages write.
    message = await persist_envelope(
        db,
        conversation_id=conversation.id,
        sender_id=sender.id,
        ciphertext=ciphertext,
        nonce=nonce,
        key_epoch=envelope.key_epoch,
    )
    return serialize_envelope(message)


# Confirm the socket's user may join this conversation and that both keys exist.
async def authorize_relay_connection(
    db: AsyncSession,
    conversation_id: UUID,
    user: User,
) -> Conversation:
    """Return the conversation after membership and both-parties-have-keys checks."""

    conversation = await get_conversation_for_participant(db, conversation_id, user)
    peer_user = await db.get(User, other_user_id(conversation, user.id))
    if peer_user is None:
        raise EnvelopeRejected("conversation peer is missing")
    # Slice 4 message endpoints must reject either party with public_key is None.
    require_both_public_keys(user, peer_user)
    return conversation
