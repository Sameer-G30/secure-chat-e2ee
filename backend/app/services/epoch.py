"""Coordinate the non-secret per-conversation epoch counter. Never derive a key."""

# Import timezone-aware clocks for the 24h half of the rotation rule.
from datetime import UTC, datetime, timedelta

# Import UUID only through the conversation row; keep this module ciphertext-free.
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Import the conversation row that holds current_epoch and last_rotated_at.
from app.models.conversation import Conversation

# Import Message so the message-count trigger stays conversation-scoped.
from app.models.message import Message

# Import the shared naive-to-UTC normalizer (also used by app/routers/auth.py) instead
# of keeping a second, byte-for-byte identical copy in this service.
from app.utils.time import as_utc


# Decide whether this persist should increment current_epoch.
def rotation_is_due(
    conversation: Conversation,
    messages_since_last_bump: int,
    *,
    rotate_after_messages: int,
    rotate_after_hours: int,
    now: datetime,
) -> bool:
    """Return True when the documented N-messages OR 24h rule fires.

    Neither trigger rotates every message when N > 1. A value of 0 disables
    that half of the rule so tests can isolate the other half.
    """

    # Message-count half: N envelopes since last_rotated_at (or since the start).
    if rotate_after_messages > 0 and messages_since_last_bump >= rotate_after_messages:
        # The just-persisted envelope is included in the count.
        return True
    # Wall-clock half: 24h (or configured hours) since last bump or created_at.
    if rotate_after_hours > 0:
        # NULL last_rotated_at means this conversation has never been bumped.
        anchor = conversation.last_rotated_at or conversation.created_at
        # Compare in UTC so SQLite naive timestamps still work.
        elapsed = now - as_utc(anchor)
        # Fire when the idle/active window is at least the configured hours.
        if elapsed >= timedelta(hours=rotate_after_hours):
            return True
    # Neither half fired; keep the current subkey id.
    return False


# Count persisted envelopes in this conversation since the last bump (or all, if never bumped).
async def count_messages_since_last_bump(
    db: AsyncSession,
    conversation: Conversation,
) -> int:
    """Return a conversation-scoped envelope count used only as a rotation trigger.

    The query is always filtered by conversation_id (§2, §11). The server
    still does not inspect ciphertext bytes.
    """

    # Start from a conversation-scoped count, never the whole messages table.
    filters = [Message.conversation_id == conversation.id]
    # After a bump, only envelopes newer than last_rotated_at count toward N.
    if conversation.last_rotated_at is not None:
        # Strict greater-than so the rotating persist is not immediately re-counted.
        filters.append(Message.created_at > conversation.last_rotated_at)
    # Scalar COUNT(*) over the filtered conversation rows.
    counted = await db.scalar(select(func.count()).select_from(Message).where(*filters))
    # SQLAlchemy types COUNT as int | None; treat a missing scalar as zero.
    return int(counted or 0)


# Atomically increment conversations.current_epoch and stamp last_rotated_at.
async def increment_current_epoch(
    db: AsyncSession,
    conversation: Conversation,
    *,
    now: datetime,
) -> int:
    """Persist current_epoch = current_epoch + 1 and return the new integer.

    The UPDATE is a SQL increment so two concurrent bumps cannot clobber
    each other back to the same value. This still only stores a counter,
    never a key.
    """

    # Increment and stamp in one statement; RETURNING gives the new counter.
    new_epoch = await db.scalar(
        update(Conversation)
        .where(Conversation.id == conversation.id)
        .values(
            current_epoch=Conversation.current_epoch + 1,
            last_rotated_at=now,
            messages_in_current_epoch=0,
        )
        .returning(Conversation.current_epoch)
    )
    # A missing RETURNING would mean the conversation row disappeared; fail closed.
    if new_epoch is None:
        # Keep the error free of envelope bytes or key material.
        raise RuntimeError("epoch increment did not return a counter")
    # Persist the bump before broadcasting so GET .../epoch agrees with the WS frame.
    await db.commit()
    # Keep the in-memory conversation used by this WebSocket in sync.
    conversation.current_epoch = int(new_epoch)
    conversation.last_rotated_at = now
    conversation.messages_in_current_epoch = 0
    # Return the integer clients will put in the next envelope's key_epoch.
    return int(new_epoch)


# After a qualifying persist, maybe bump the counter; return the new epoch or None.
async def maybe_rotate_epoch(
    db: AsyncSession,
    conversation: Conversation,
    *,
    rotate_after_messages: int,
    rotate_after_hours: int,
    now: datetime | None = None,
    persisted_message: Message | None = None,
) -> int | None:
    """Increment current_epoch when N messages OR 24h since last bump.

    Callers must already have persisted the qualifying envelope. Returns
    the new current_epoch when a bump happened, otherwise None. The server
    never generates a key — only this integer.

    messages_in_current_epoch is the O(1) replacement for COUNT(*) of rows
    with created_at > last_rotated_at. A persist whose created_at is not
    strictly after last_rotated_at does not increment (SQLite stores
    timestamps at second precision, so same-second follow-up persists must
    not immediately re-trigger N).
    """

    # Use an explicit clock so tests can freeze "now" for the 24h rule.
    clock = now if now is not None else datetime.now(UTC)
    # Match COUNT(*) WHERE created_at > last_rotated_at without scanning messages.
    counts_toward_n = True
    # After a bump, only later envelopes count toward the next N.
    if conversation.last_rotated_at is not None and persisted_message is not None:
        # A missing created_at cannot be compared; count the persist (fail toward N).
        created_at = persisted_message.created_at
        if created_at is not None and as_utc(created_at) <= as_utc(conversation.last_rotated_at):
            # Same second as the bump stamp: do not increment the O(1) counter.
            counts_toward_n = False
    # Increment only when this persist would have been included in the old COUNT.
    if counts_toward_n:
        # O(1) increment; never COUNT(*) the messages table here.
        # Read the stored counter; treat NULL/missing as zero.
        prior = int(conversation.messages_in_current_epoch or 0)
        # Persist the O(1) increment; never COUNT(*) the messages table here.
        conversation.messages_in_current_epoch = prior + 1
        # Flush so a concurrent persist on another socket sees the new value.
        await db.flush()
    # Apply the documented OR rule; do not rotate on every persist when N > 1.
    if not rotation_is_due(
        conversation,
        conversation.messages_in_current_epoch,
        rotate_after_messages=rotate_after_messages,
        rotate_after_hours=rotate_after_hours,
        now=clock,
    ):
        # Persist the counter increment even when the epoch does not bump.
        await db.commit()
        return None
    # Persist the atomic increment, stamp last_rotated_at, and reset the counter.
    return await increment_current_epoch(db, conversation, now=clock)
