"""Create, remove, and check server-enforced blocks.

Unlike the legacy React prototype's block feature (`localStorage` only, never
synced, never enforced — a blocked user could still send via Firebase Realtime
Database), this is checked at the two points that actually matter
cryptographically: opening the relay WebSocket (`authorize_relay_connection`
in app/services/relay.py) and, defensively, on every persisted envelope.
Blocking does *not* prevent adding someone as a contact or starting a
conversation row — those are harmless metadata, and hiding their existence
would itself leak information about who blocked whom.
"""

# Import UUID for typed account-identifier parameters.
from uuid import UUID

# Import FastAPI's HTTP-error primitives so callers can return stable status codes.
from fastapi import HTTPException, status

# Import SQLAlchemy query helpers, the uniqueness-conflict error, and the OR combinator.
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM models this service reads and writes.
from app.models.block import Block
from app.models.user import User

# Import the response shapes so routers do not rebuild block payloads.
from app.schemas.blocks import BlockListResponse, BlockResponse

# Shared detail when the named account does not exist.
_USER_NOT_FOUND_DETAIL = "user not found"
# Shared detail when the caller tries to block their own account.
_SELF_BLOCK_DETAIL = "cannot block yourself"


# Build the client-facing block payload from an ORM row plus the blocked user.
def serialize_block(block: Block, blocked_user: User) -> BlockResponse:
    """Return the blocked account's id, handle, and when the block was created."""

    return BlockResponse(
        id=blocked_user.id, username=blocked_user.username, created_at=block.created_at
    )


# Return True if either account has blocked the other, in either direction.
async def is_blocked_either_direction(db: AsyncSession, user_a_id: UUID, user_b_id: UUID) -> bool:
    """Check both directions so a block is symmetric in its effect on delivery.

    The relay must refuse to connect (or deliver) regardless of who blocked
    whom — a blocker should not receive the blocked party's messages, and a
    blocked party should not be able to keep messaging someone who blocked them.
    """

    existing = await db.scalar(
        select(Block.id).where(
            or_(
                and_(Block.blocker_id == user_a_id, Block.blocked_id == user_b_id),
                and_(Block.blocker_id == user_b_id, Block.blocked_id == user_a_id),
            )
        )
    )
    return existing is not None


# Return the caller's full block list, newest first.
async def list_blocks_for_owner(db: AsyncSession, owner: User) -> BlockListResponse:
    """Load every account the caller has blocked, joined to usernames."""

    rows = (
        await db.execute(
            select(Block, User)
            .join(User, User.id == Block.blocked_id)
            .where(Block.blocker_id == owner.id)
            .order_by(Block.created_at.desc())
        )
    ).all()
    items = [serialize_block(block, blocked_user) for block, blocked_user in rows]
    return BlockListResponse(blocks=items)


# Block a named account, idempotently.
async def block_user_for_owner(db: AsyncSession, owner: User, username: str) -> BlockResponse:
    """Insert (blocker_id, blocked_id) or return the existing row.

    Unknown usernames 404. Self-block 400. A duplicate block is 200 with the
    existing row so the UI can retry safely.
    """

    if username == owner.username:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=_SELF_BLOCK_DETAIL)

    blocked_user = await db.scalar(select(User).where(User.username == username))
    if blocked_user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_USER_NOT_FOUND_DETAIL)

    existing = await db.scalar(
        select(Block).where(Block.blocker_id == owner.id, Block.blocked_id == blocked_user.id)
    )
    if existing is not None:
        return serialize_block(existing, blocked_user)

    block = Block(blocker_id=owner.id, blocked_id=blocked_user.id)
    db.add(block)
    try:
        await db.commit()
    except IntegrityError:
        # Another request created the same edge first; load that winner instead of 500ing.
        await db.rollback()
        existing = await db.scalar(
            select(Block).where(Block.blocker_id == owner.id, Block.blocked_id == blocked_user.id)
        )
        if existing is None:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, detail="could not save block"
            ) from None
        return serialize_block(existing, blocked_user)

    await db.refresh(block)
    return serialize_block(block, blocked_user)


# Remove a block, idempotently (unblocking a non-blocked account is a no-op).
async def unblock_user_for_owner(db: AsyncSession, owner: User, username: str) -> None:
    """Delete (blocker_id, blocked_id) if present; never errors on "not blocked."""

    blocked_user = await db.scalar(select(User).where(User.username == username))
    if blocked_user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_USER_NOT_FOUND_DETAIL)

    existing = await db.scalar(
        select(Block).where(Block.blocker_id == owner.id, Block.blocked_id == blocked_user.id)
    )
    if existing is not None:
        await db.delete(existing)
        await db.commit()
