"""Define the server-enforced block edge: legacy parity would have been local-only.

The legacy React prototype's "block" feature lived entirely in `localStorage` — it
never reached the server, so a blocked user could still send messages via Firebase
Realtime Database; only the blocking user's own browser hid them. This table makes
blocking an actual security property enforced at the relay (see
`app/services/relay.py` and `app/routers/ws.py`), not just a client-side filter.
"""

# Import datetime for typed created_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import constraint helpers, timestamps, foreign keys, and the portable UUID type.
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, UniqueConstraint, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


class Block(Base):
    """Represent one account blocking another. Metadata only: who blocked whom, and when.

    Enforcement lives in the relay layer (WebSocket connect is refused, and a
    message from a blocked sender is silently dropped rather than delivered)
    rather than at contact-list or conversation-creation time, so blocking a
    user does not retroactively hide that a conversation with them exists.
    """

    __tablename__ = "blocks"

    # Identify each block edge with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify the account that created this block.
    blocker_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Identify the account being blocked.
    blocked_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Record when the block was created using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # One row per (blocker, blocked) pair so blocking twice is idempotent.
        UniqueConstraint("blocker_id", "blocked_id", name="uq_blocks_blocker_blocked"),
        # An account cannot block itself.
        CheckConstraint("blocker_id != blocked_id", name="ck_blocks_not_self"),
    )
