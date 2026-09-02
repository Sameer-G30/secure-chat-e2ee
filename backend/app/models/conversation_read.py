"""Define the per-member read cursor used for unread badges.

This is metadata only: a timestamp (and optional message id) recording when
this account last focused a conversation. The server never stores a preview
string or any decrypted body.
"""

# Import datetime for typed last_read_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import constraint helpers, timestamps, and foreign keys.
from sqlalchemy import DateTime, ForeignKey, UniqueConstraint, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


class ConversationRead(Base):
    """Represent one member's last-read cursor in one conversation."""

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "conversation_reads"

    # Identify each cursor row with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify the account whose unread badge this cursor drives.
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Identify the conversation this cursor belongs to; every count filters on this.
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Record when this member last marked the conversation read.
    last_read_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Optionally pin the newest envelope this member had focused; NULL means time-only.
    last_read_message_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )

    # One cursor per (member, conversation); marking read upserts this row.
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "conversation_id",
            name="uq_conversation_reads_user_conversation",
        ),
    )
