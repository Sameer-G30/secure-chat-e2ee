"""Define delivered/read markers the peer's device reports for one envelope.

These timestamps are metadata the server can already infer from WebSocket
activity (the client was online and later focused the chat). They never
include plaintext, previews, or keys.
"""

# Import datetime for typed delivered_at / read_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import constraint helpers, timestamps, and foreign keys.
from sqlalchemy import DateTime, ForeignKey, UniqueConstraint, Uuid

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


class MessageReceipt(Base):
    """Represent one recipient's delivered/read state for one stored envelope."""

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "message_receipts"

    # Identify each receipt row with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify the envelope these ticks describe. Cascades with hard delete-for-everyone.
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Identify the member who received (not sent) the envelope.
    recipient_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Record when the recipient's device acknowledged the ciphertext (NULL = not yet).
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Record when the recipient focused the chat on that envelope (NULL = not yet).
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # One receipt row per (envelope, recipient); delivered then read update in place.
    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "recipient_id",
            name="uq_message_receipts_message_recipient",
        ),
    )
