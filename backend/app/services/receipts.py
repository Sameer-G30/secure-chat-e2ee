"""Record delivered/read ticks and last-read cursors without storing a message body."""

# Import datetime helpers so last-read and receipt stamps are timezone-aware.
from datetime import UTC, datetime

# Import UUID for typed conversation, member, and envelope identifiers.
from uuid import UUID

# Import FastAPI's HTTP-error primitives so unknown kinds become 400s.
from fastapi import HTTPException, status

# Import SQLAlchemy query helpers used for conversation-scoped reads.
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM models this service reads and writes.
from app.models.conversation import Conversation
from app.models.conversation_read import ConversationRead
from app.models.message import Message
from app.models.message_hide import MessageHide
from app.models.message_receipt import MessageReceipt
from app.models.user import User


# Stamp a UTC now() used for last-read and receipt columns.
def _utc_now() -> datetime:
    """Return the current UTC time for receipt and last-read columns."""

    return datetime.now(UTC)


# Load or create the caller's last-read cursor for one conversation.
async def upsert_conversation_read(
    db: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
    last_read_message_id: UUID | None,
    at: datetime,
) -> ConversationRead:
    """Insert or update this member's last-read cursor; never stores a preview."""

    # Look up the existing cursor for this (member, conversation) pair.
    existing = await db.scalar(
        select(ConversationRead).where(
            ConversationRead.user_id == user_id,
            ConversationRead.conversation_id == conversation_id,
        )
    )
    if existing is None:
        # First open of this chat: insert a new cursor row.
        existing = ConversationRead(
            user_id=user_id,
            conversation_id=conversation_id,
            last_read_at=at,
            last_read_message_id=last_read_message_id,
        )
        db.add(existing)
    else:
        # Later opens move the cursor forward to the latest focus time.
        existing.last_read_at = at
        existing.last_read_message_id = last_read_message_id
    return existing


# Load or create one recipient's receipt row for one stored envelope.
async def upsert_message_receipt(
    db: AsyncSession,
    *,
    message_id: UUID,
    recipient_id: UUID,
    kind: str,
    at: datetime,
) -> MessageReceipt:
    """Set delivered_at and/or read_at on this recipient's receipt row.

    kind 'read' also sets delivered_at when it was still empty.
    """

    # Look up the existing receipt for this (envelope, recipient) pair.
    existing = await db.scalar(
        select(MessageReceipt).where(
            MessageReceipt.message_id == message_id,
            MessageReceipt.recipient_id == recipient_id,
        )
    )
    if existing is None:
        # First ack: insert a new receipt row with the requested timestamps.
        existing = MessageReceipt(
            message_id=message_id,
            recipient_id=recipient_id,
            delivered_at=at,
            read_at=at if kind == "read" else None,
        )
        db.add(existing)
        return existing
    if existing.delivered_at is None:
        # A later delivered or read ack fills in the missing delivered stamp.
        existing.delivered_at = at
    if kind == "read" and existing.read_at is None:
        # A read ack fills in the missing read stamp; already-read stays first-read.
        existing.read_at = at
    return existing


# Apply one WebSocket receipt batch and return frames to fan out to each sender.
async def apply_receipt_batch(
    db: AsyncSession,
    *,
    conversation: Conversation,
    recipient: User,
    kind: str,
    message_ids: list[UUID],
) -> list[dict[str, str]]:
    """Record delivered/read ticks for envelopes this member received.

    Skips unknown ids, own-sent envelopes, and envelopes outside this
    conversation. Returns metadata frames the WebSocket loop broadcasts to
    the original sender. Never includes ciphertext or a message body.
    """

    if kind not in {"delivered", "read"}:
        # A malformed kind must not write a receipt row.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid receipt kind")

    # Stamp every row in this batch with the same server clock reading.
    at = _utc_now()
    # Collect outbound metadata frames for senders who still have a socket.
    outbound: list[dict[str, str]] = []
    # Deduplicate ids so a retried frame cannot double-write in one batch.
    seen: set[UUID] = set()
    for message_id in message_ids:
        if message_id in seen:
            # Ignore a duplicate id in the same batch.
            continue
        seen.add(message_id)
        # Scope the lookup by conversation_id so a guessed UUID from another chat is ignored.
        stored = await db.scalar(
            select(Message).where(
                Message.id == message_id,
                Message.conversation_id == conversation.id,
            )
        )
        if stored is None:
            # Unknown or foreign envelope: skip rather than failing the whole batch.
            continue
        if stored.sender_id == recipient.id:
            # A sender cannot ack their own envelope.
            continue
        await upsert_message_receipt(
            db,
            message_id=stored.id,
            recipient_id=recipient.id,
            kind=kind,
            at=at,
        )
        # Tell the original sender that this recipient's device has ticked the envelope.
        outbound.append(
            {
                "type": "receipt",
                "kind": kind,
                "message_id": str(stored.id),
                "recipient_id": str(recipient.id),
            }
        )
    await db.commit()
    return outbound


# Mark the conversation read and ack every still-unacked inbound envelope.
async def mark_conversation_read(
    db: AsyncSession,
    *,
    conversation: Conversation,
    viewer: User,
) -> tuple[datetime, list[dict[str, str]]]:
    """Upsert the last-read cursor and mark inbound envelopes delivered+read.

    Returns (last_read_at, receipt frames to broadcast to the peer).
    """

    at = _utc_now()
    # Find inbound envelopes in this conversation that the viewer has not read yet.
    hidden_ids = select(MessageHide.message_id).where(MessageHide.owner_id == viewer.id)
    inbound = (
        await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .where(Message.sender_id != viewer.id)
            .where(Message.id.not_in(hidden_ids))
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
    ).all()
    # Pin the cursor to the newest inbound envelope when any exist.
    last_message_id = inbound[-1].id if inbound else None
    await upsert_conversation_read(
        db,
        user_id=viewer.id,
        conversation_id=conversation.id,
        last_read_message_id=last_message_id,
        at=at,
    )
    outbound: list[dict[str, str]] = []
    for stored in inbound:
        receipt = await upsert_message_receipt(
            db,
            message_id=stored.id,
            recipient_id=viewer.id,
            kind="read",
            at=at,
        )
        if receipt.read_at == at:
            # Only fan out when this call actually advanced the read stamp.
            outbound.append(
                {
                    "type": "receipt",
                    "kind": "read",
                    "message_id": str(stored.id),
                    "recipient_id": str(viewer.id),
                }
            )
    await db.commit()
    return at, outbound


# Map the viewer's own sent envelope ids onto the peer's delivered/read flags.
async def peer_receipt_flags(
    db: AsyncSession,
    *,
    peer_id: UUID,
    message_ids: list[UUID],
) -> dict[UUID, tuple[bool, bool]]:
    """Return {message_id: (delivered, read)} for the peer's receipts.

    Missing rows mean the peer has not acknowledged yet. This never reads
    Message rows (receipts already point at specific ids from a conversation-
    scoped history query).
    """

    if not message_ids:
        return {}
    rows = (
        await db.scalars(
            select(MessageReceipt).where(
                MessageReceipt.recipient_id == peer_id,
                MessageReceipt.message_id.in_(message_ids),
            )
        )
    ).all()
    return {
        row.message_id: (row.delivered_at is not None, row.read_at is not None) for row in rows
    }


# Count unread inbound envelopes for each named peer on the owner's address book.
async def unread_counts_for_owner(
    db: AsyncSession,
    owner: User,
    peer_ids: list[UUID],
) -> dict[UUID, int]:
    """Return {peer_id: unread_count} using last-read cursors, never a preview.

    Unread means: a 1:1 conversation exists, the envelope was sent by the peer,
    it is newer than this owner's last_read_at (or there is no cursor), and
    this owner has not hidden it. Zero when there is no conversation yet.
    """

    counts: dict[UUID, int] = {peer_id: 0 for peer_id in peer_ids}
    if not peer_ids:
        return counts
    # Load every 1:1 conversation the owner shares with any listed peer.
    conversations = (
        await db.scalars(
            select(Conversation).where(
                (Conversation.user_a_id == owner.id) | (Conversation.user_b_id == owner.id)
            )
        )
    ).all()
    # Index conversations by the peer id on the other side of the pair.
    by_peer: dict[UUID, Conversation] = {}
    for conversation in conversations:
        peer_id = (
            conversation.user_b_id
            if conversation.user_a_id == owner.id
            else conversation.user_a_id
        )
        if peer_id in counts:
            by_peer[peer_id] = conversation
    if not by_peer:
        return counts
    conversation_ids = [conversation.id for conversation in by_peer.values()]
    # Load this owner's last-read cursors for those conversations.
    cursors = (
        await db.scalars(
            select(ConversationRead).where(
                ConversationRead.user_id == owner.id,
                ConversationRead.conversation_id.in_(conversation_ids),
            )
        )
    ).all()
    last_read_at_by_conversation = {row.conversation_id: row.last_read_at for row in cursors}
    hidden_ids = select(MessageHide.message_id).where(MessageHide.owner_id == owner.id)
    for peer_id, conversation in by_peer.items():
        filters = [
            Message.conversation_id == conversation.id,
            Message.sender_id == peer_id,
            Message.id.not_in(hidden_ids),
        ]
        last_read_at = last_read_at_by_conversation.get(conversation.id)
        if last_read_at is not None:
            filters.append(Message.created_at > last_read_at)
        unread = await db.scalar(select(func.count()).select_from(Message).where(*filters))
        counts[peer_id] = int(unread or 0)
    return counts
