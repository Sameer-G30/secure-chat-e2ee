"""Define the per-owner "delete for me" marker.

The legacy React prototype's delete-for-me wrote `deleted: true` directly onto
the shared message row (and had a field-name bug: the UI checked `isDeleted`
instead) — meaning "delete for me" was actually implemented as "hide this from
everyone," the same bug class as its `edited`/`isEdited` mismatch. This table
fixes that by scoping the hide to the requesting owner only: the peer's copy of
the same envelope is completely unaffected.
"""

# Import datetime for typed created_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import constraint helpers, timestamps, and foreign keys.
from sqlalchemy import DateTime, ForeignKey, UniqueConstraint, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


class MessageHide(Base):
    """Represent one owner's decision to hide one envelope from their own history only.

    The peer's copy is unaffected; this is "delete for me," not "delete for
    everyone" (see `messages.edited_at`/hard-delete for that feature instead).
    """

    __tablename__ = "message_hides"

    # Identify each hide row with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify the account whose history view excludes this message.
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Identify the hidden envelope. Cascades so a hard delete-for-everyone (which
    # removes the Message row outright) does not leave an orphaned hide marker.
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Record when the owner hid this message using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # Hiding the same message twice is idempotent, not two rows.
        UniqueConstraint("owner_id", "message_id", name="uq_message_hides_owner_message"),
    )
