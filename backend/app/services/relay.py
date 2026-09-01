"""Persist and fan out ciphertext envelopes without ever decrypting them."""

# Import base64 to decode inbound wire format and re-encode outbound frames.
import base64

# Import the shared UTC clock helper used to stamp edited_at.
from datetime import UTC, datetime

# Import UUID for typed conversation and sender identifiers.
from uuid import UUID, uuid4

# Import FastAPI's WebSocket type and HTTP-error primitives (the latter for the REST
# delete/hide endpoints this module also backs, alongside the WebSocket relay).
from fastapi import HTTPException, WebSocket, status

# Import SQLAlchemy query helpers used for conversation-scoped reads.
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

# Import validated settings so rotation thresholds are read at persist time.
from app.config import get_settings

# Import the ORM models this service reads and writes.
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.message_hide import MessageHide
from app.models.user import User

# Import the validated inbound envelope and the outbound relay frame.
from app.schemas.messages import RelayEnvelopeIn, RelayEnvelopeOut

# Import server-enforced blocking so a blocked sender's envelope is dropped, not relayed.
from app.services.blocks import is_blocked_either_direction

# Import conversation membership and public-key enforcement helpers.
from app.services.conversations import (
    get_conversation_for_participant,
    other_user_id,
    require_both_public_keys,
)

# Import Slice 8 epoch-counter rotation; this module still never derives a key.
from app.services.epoch import maybe_rotate_epoch

# Shared detail when a REST caller tries to delete/act on a message they did not send.
_NOT_SENDER_DETAIL = "only the sender may delete this message"
# Shared detail when a REST caller references a message outside their own conversation.
_MESSAGE_NOT_FOUND_DETAIL = "message not found"


# Raised when an envelope's routing metadata does not match the authenticated context.
class EnvelopeRejected(Exception):
    """Reject a frame at the protocol layer without attempting any cryptographic verify."""


# Raised when a sender's own envelope was accepted but silently not delivered because
# the recipient has blocked them. The sender still gets an "accepted" ack (so the
# composer does not show a stuck-sending state and the block is not disclosed), but
# the peer never sees it and it is never persisted.
class EnvelopeSilentlyDropped(Exception):
    """Signal a fake-accepted, never-persisted, never-broadcast envelope."""

    def __init__(self, fake_id: UUID) -> None:
        # Carry a locally generated id the sender's "accepted" frame can reference.
        # It never corresponds to a stored row.
        self.fake_id = fake_id
        super().__init__("envelope silently dropped: recipient has blocked the sender")


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

    def is_connected(self, conversation_id: UUID, user_id: UUID) -> bool:
        """Return whether this user currently has a live socket in the room."""

        # Presence is metadata the server can already see; this is not a secret.
        room = self._rooms.get(conversation_id, {})
        sockets = room.get(user_id)
        return bool(sockets)

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
                try:
                    # Send to one member; a dead socket must not break the whole broadcast.
                    await websocket.send_json(payload)
                except Exception:
                    # The socket is gone (client dropped without a clean close frame, or the
                    # send raced a half-closed connection). Evict it immediately so later
                    # broadcasts and is_connected() checks stop trying to use it, then keep
                    # fanning out to every other still-live member in this room.
                    self.leave(conversation_id, user_id, websocket)


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
    ad_version: int = 1,
    client_message_id: UUID | None = None,
    revision: int = 0,
) -> Message:
    """Insert one envelope row. Callers must already have authorized membership.

    The INSERT always includes conversation_id so a later SELECT scoped by
    conversation_id can never return another conversation's rows. ad_version
    defaults to 1 (the original, message-identity-free associated data); a
    client that sends `message_id` gets ad_version=2 and a stored
    client_message_id, enabling a later edit (see find_editable_message /
    apply_edit below) without changing how any existing v1 row decrypts.
    """

    message = Message(
        conversation_id=conversation_id,
        sender_id=sender_id,
        ciphertext=ciphertext,
        nonce=nonce,
        key_epoch=key_epoch,
        ad_version=ad_version,
        client_message_id=client_message_id,
        revision=revision,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


# Look up an existing v2 message this sender may edit within one conversation.
async def find_editable_message(
    db: AsyncSession,
    conversation_id: UUID,
    client_message_id: UUID,
    sender_id: UUID,
) -> Message | None:
    """Return the row a resend with this `message_id` would be editing, if any.

    Scoped by conversation_id (§2, §11) *and* sender_id: only the original
    sender may ever edit their own message. A different sender presenting the
    same client_message_id (accidentally, or a malicious client deliberately
    colliding with someone else's id) simply never matches here and is
    treated as an unrelated new message instead — it can never overwrite
    another sender's row, because this query would return None for them.
    """

    result: Message | None = await db.scalar(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.client_message_id == client_message_id,
            Message.sender_id == sender_id,
        )
    )
    return result


# Apply a validated edit to an existing v2 message row.
async def apply_edit(
    db: AsyncSession,
    message: Message,
    *,
    ciphertext: bytes,
    nonce: bytes,
    key_epoch: int,
    revision: int,
) -> Message:
    """Overwrite ciphertext/nonce/key_epoch/revision and stamp edited_at.

    Callers must already have confirmed `revision == message.revision + 1`
    (see relay_envelope) — that check, not this function, is what closes the
    edit-rollback gap: the server can only ever move a message's stored
    revision forward by exactly one, never backward and never sideways.
    `created_at` is left untouched, so epoch-rotation's message count (which
    counts rows by `created_at`, not by write) is unaffected by an edit.
    """

    message.ciphertext = ciphertext
    message.nonce = nonce
    message.key_epoch = key_epoch
    message.revision = revision
    message.edited_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(message)
    return message


# Load a message the caller sent, scoped by conversation, for a REST delete.
async def get_message_for_sender(
    db: AsyncSession,
    conversation_id: UUID,
    message_id: UUID,
    sender_id: UUID,
) -> Message:
    """Return the row, or 404/403 without ever revealing whether it exists to a non-sender.

    Scoped by conversation_id first (§2, §11): a message id from a different
    conversation the caller is a member of must still 404, not 403.
    """

    message = await db.scalar(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.id == message_id,
        )
    )
    if message is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_MESSAGE_NOT_FOUND_DETAIL)
    if message.sender_id != sender_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=_NOT_SENDER_DETAIL)
    return message


# Hard-delete one message row ("delete for everyone"), sender-only.
async def delete_message_for_everyone(db: AsyncSession, message: Message) -> None:
    """Remove the row outright; there is no soft-delete/tombstone for this feature.

    A hard delete (rather than the legacy app's buggy `deleted: true` flag,
    which the UI never actually read correctly — see docs/02-legacy-feature-inventory.md)
    means a later `GET .../messages` simply cannot return this row again, and any
    `message_hides` marker pointing at it cascades away with it (see
    app/models/message_hide.py's ON DELETE CASCADE).
    """

    await db.delete(message)
    await db.commit()


# Hide one message from exactly one owner's own history ("delete for me").
async def hide_message_for_owner(db: AsyncSession, message: Message, owner_id: UUID) -> None:
    """Insert a MessageHide row, idempotently; the peer's copy is never touched.

    Unlike the legacy app's delete-for-me (a shared `deleted` field on the
    message row, which the UI read as `isDeleted` and so never displayed
    correctly for *either* party), this only ever changes what the calling
    owner's own `GET .../messages` returns.
    """

    existing = await db.scalar(
        select(MessageHide).where(
            MessageHide.owner_id == owner_id,
            MessageHide.message_id == message.id,
        )
    )
    if existing is not None:
        return
    db.add(MessageHide(owner_id=owner_id, message_id=message.id))
    await db.commit()


# Read envelopes for exactly one conversation; never the whole messages table.
async def list_envelopes_for_conversation(
    db: AsyncSession,
    conversation_id: UUID,
    *,
    newest_first: bool = True,
    exclude_hidden_for: UUID | None = None,
    limit: int | None = None,
    after_id: UUID | None = None,
) -> list[Message]:
    """Return stored envelopes for one conversation_id.

    Defaults to newest first. Slice 7's history GET passes newest_first=False
    so the transcript can render oldest-to-newest. id is the tiebreaker when
    timestamps match. Every read stays conversation-scoped (§2, §11).

    `exclude_hidden_for`, when given a user id, drops rows that user has
    hidden with "delete for me" (app/models/message_hide.py). This is
    per-caller: the peer's own `GET .../messages` call is unaffected, since it
    passes its own id (or none) to this same parameter.

    Optional `limit`/`after_id` implement cursor pagination without changing
    the default "return everything" behavior existing clients rely on.
    """

    rows, _next_cursor = await list_envelopes_page(
        db,
        conversation_id,
        newest_first=newest_first,
        exclude_hidden_for=exclude_hidden_for,
        limit=limit,
        after_id=after_id,
    )
    return rows


# Page conversation history and report whether another page exists.
async def list_envelopes_page(
    db: AsyncSession,
    conversation_id: UUID,
    *,
    newest_first: bool = True,
    exclude_hidden_for: UUID | None = None,
    limit: int | None = None,
    after_id: UUID | None = None,
) -> tuple[list[Message], UUID | None]:
    """Return (rows, next_cursor). next_cursor is the last row's id when more remain."""

    created_order = Message.created_at.desc() if newest_first else Message.created_at.asc()
    id_order = Message.id.desc() if newest_first else Message.id.asc()
    query = select(Message).where(Message.conversation_id == conversation_id)
    if exclude_hidden_for is not None:
        hidden_ids = select(MessageHide.message_id).where(
            MessageHide.owner_id == exclude_hidden_for
        )
        query = query.where(Message.id.not_in(hidden_ids))
    if after_id is not None:
        cursor = await db.get(Message, after_id)
        if cursor is None or cursor.conversation_id != conversation_id:
            return [], None
        if newest_first:
            query = query.where(
                or_(
                    Message.created_at < cursor.created_at,
                    and_(Message.created_at == cursor.created_at, Message.id < cursor.id),
                )
            )
        else:
            query = query.where(
                or_(
                    Message.created_at > cursor.created_at,
                    and_(Message.created_at == cursor.created_at, Message.id > cursor.id),
                )
            )
    if limit is not None:
        query = query.limit(limit + 1)
    result = await db.scalars(query.order_by(created_order, id_order))
    rows = list(result.all())
    next_cursor: UUID | None = None
    if limit is not None and len(rows) > limit:
        rows = rows[:limit]
        next_cursor = rows[-1].id
    return rows, next_cursor


# Serialize a stored envelope for the WebSocket without adding plaintext fields.
def serialize_envelope(message: Message) -> RelayEnvelopeOut:
    """Return the ciphertext-only frame the peer will decrypt locally."""

    created_at = message.created_at.isoformat() if message.created_at is not None else ""
    edited_at = message.edited_at.isoformat() if message.edited_at is not None else None
    return RelayEnvelopeOut(
        id=message.id,
        conversation_id=message.conversation_id,
        sender_id=message.sender_id,
        ciphertext=base64.b64encode(message.ciphertext).decode("ascii"),
        nonce=base64.b64encode(message.nonce).decode("ascii"),
        key_epoch=message.key_epoch,
        created_at=created_at,
        ad_version=message.ad_version,
        message_id=message.client_message_id,
        revision=message.revision,
        edited_at=edited_at,
    )


# Authorize, persist, and fan out one inbound WebSocket envelope.
async def relay_envelope(
    db: AsyncSession,
    *,
    conversation: Conversation,
    sender: User,
    envelope: RelayEnvelopeIn,
) -> tuple[RelayEnvelopeOut, int | None]:
    """Validate routing, store ciphertext only, and maybe bump current_epoch.

    Returns (outbound envelope, new current_epoch or None). The server never
    decrypts. Associated-data verification is the client's job. A peer socket
    may have already bumped the counter, so current_epoch is refreshed first.

    Three outcomes, in order:
    1. The recipient has blocked the sender (or vice versa) — raises
       EnvelopeSilentlyDropped so the caller can fake-ack the sender without
       persisting or delivering anything (see app/routers/ws.py).
    2. `envelope.message_id` matches an existing row this sender owns — this
       is an edit: the row is overwritten in place, revision must advance by
       exactly one, and the epoch-rotation message count is untouched.
    3. Otherwise — a new message, ad_version=2 if `message_id` was supplied
       (enabling a future edit), ad_version=1 if not (unchanged legacy path).
    """

    # Reload membership + epoch so a bump on the other socket is visible here.
    await db.refresh(conversation)
    # Reject claimed conversation/sender/epoch that do not match this connection.
    assert_envelope_routing(envelope, conversation.id, sender.id, conversation.current_epoch)

    peer_id = other_user_id(conversation, sender.id)
    if await is_blocked_either_direction(db, sender.id, peer_id):
        # Never persist or deliver; the sender still gets a fake, uncorrelated
        # "accepted" ack so their composer does not show a stuck-sending state
        # and the existence of a block is never disclosed to either party.
        raise EnvelopeSilentlyDropped(uuid4())

    # Decode wire base64 into BYTEA columns; lengths were already validated.
    ciphertext, nonce = decode_envelope_bytes(envelope)

    if envelope.message_id is not None:
        existing = await find_editable_message(db, conversation.id, envelope.message_id, sender.id)
        if existing is not None:
            if envelope.revision != existing.revision + 1:
                # The server can only ever move a stored revision forward by exactly
                # one; anything else is either a stale retry or a rollback attempt.
                raise EnvelopeRejected(
                    "revision must be exactly one greater than the current revision"
                )
            edited = await apply_edit(
                db,
                existing,
                ciphertext=ciphertext,
                nonce=nonce,
                key_epoch=envelope.key_epoch,
                revision=envelope.revision,
            )
            # An edit never counts toward epoch rotation and never bumps current_epoch.
            return serialize_envelope(edited), None

    # A new message: ad_version=2 (and a stored client_message_id) only when the
    # sender supplied one, so a future resend with that same id can be treated as
    # an edit; otherwise this is the unchanged ad_version=1 legacy path.
    ad_version = 2 if envelope.message_id is not None else 1
    # Insert scoped by this conversation_id; never a global messages write.
    message = await persist_envelope(
        db,
        conversation_id=conversation.id,
        sender_id=sender.id,
        ciphertext=ciphertext,
        nonce=nonce,
        key_epoch=envelope.key_epoch,
        ad_version=ad_version,
        client_message_id=envelope.message_id,
        revision=envelope.revision,
    )
    # Reload last_rotated_at in case another persist just stamped a bump.
    await db.refresh(conversation)
    # Read rotation thresholds at call time so tests can override via env.
    settings = get_settings()
    # Bump only on the documented N-messages OR 24h rule; never every send by default.
    rotated_epoch = await maybe_rotate_epoch(
        db,
        conversation,
        rotate_after_messages=settings.epoch_rotate_after_messages,
        rotate_after_hours=settings.epoch_rotate_after_hours,
        persisted_message=message,
    )
    # Return the ciphertext-only frame plus the new counter when a bump happened.
    return serialize_envelope(message), rotated_epoch


# Confirm the socket's user may join this conversation and that both keys exist.
async def authorize_relay_connection(
    db: AsyncSession,
    conversation_id: UUID,
    user: User,
) -> Conversation:
    """Return the conversation after membership, both-keys, and not-blocked checks."""

    conversation = await get_conversation_for_participant(db, conversation_id, user)
    peer_user = await db.get(User, other_user_id(conversation, user.id))
    if peer_user is None:
        raise EnvelopeRejected("conversation peer is missing")
    # Slice 4 message endpoints must reject either party with public_key is None.
    require_both_public_keys(user, peer_user)
    # Refuse the relay socket outright when either side has blocked the other,
    # rather than accepting the connection and only dropping messages later —
    # this is the strongest, simplest enforcement point for the block feature.
    if await is_blocked_either_direction(db, user.id, peer_user.id):
        raise EnvelopeRejected("conversation peer has blocked or been blocked by this account")
    return conversation
