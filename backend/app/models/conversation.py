"""Define the 1:1 conversation row the server may hold: membership and epoch only."""

# Import datetime for typed created_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import constraint helpers, timestamps, foreign keys, and the portable UUID type.
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, UniqueConstraint, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


# Map the conversations table exactly as scoped by spec §5 and plan Part B.
class Conversation(Base):
    """Represent one 1:1 conversation between two accounts.

    The server stores only membership and a non-secret epoch counter. It never
    stores session keys, epoch keys, or plaintext. Part B requires UNIQUE
    (user_a_id, user_b_id) plus CHECK (user_a_id < user_b_id) so a pair can
    only ever have one row regardless of who initiates.
    """

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "conversations"

    # Identify each conversation with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify the lexicographically-smaller participant UUID (enforced below).
    #
    # `index=True` names this `ix_conversations_user_a_id`, matching the index the
    # Alembic migration (`c4e8a2b91d07`) already creates. Before this fix the ORM
    # model and the migration had drifted apart: tests build their schema from this
    # model's metadata (`create_all`), so the test database silently lacked an index
    # a real Postgres deployment already has, and any "does the schema match the
    # migrations" assumption in this codebase was quietly false for this column.
    user_a_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # Identify the lexicographically-larger participant UUID (enforced below).
    # `index=True` names this `ix_conversations_user_b_id`, matching the migration.
    user_b_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # Store the non-secret epoch counter the clients use for KDF subkey ids.
    #
    # Slice 8 increments this integer on a documented schedule (N messages since
    # the last bump, default 50, OR 24h since last_rotated_at / created_at).
    # The server never derives or stores the corresponding key. Default 0
    # matches spec §5.
    current_epoch: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Record when the server last incremented current_epoch (NULL = never).
    #
    # Used for the 24h half of the rotation rule and to count envelopes since
    # the last bump. This is a timestamp, not a key.
    last_rotated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    # Count new envelopes persisted in the current epoch. Incremented on each
    # new send (not edits), reset to 0 when current_epoch bumps. Replaces a
    # per-persist COUNT(*) over messages so rotation stays O(1).
    messages_in_current_epoch: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Record conversation creation time using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Enforce the Part B pair-identity rules at the database layer, not only in Python.
    __table_args__ = (
        # One conversation row per unordered pair of accounts.
        UniqueConstraint("user_a_id", "user_b_id", name="uq_conversations_user_pair"),
        # Canonicalize pair order so (alice, bob) and (bob, alice) cannot both exist.
        CheckConstraint("user_a_id < user_b_id", name="ck_conversations_user_order"),
    )
