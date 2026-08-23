"""Define the server-side contact row so the UI never stores an address book in localStorage."""

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


# Map the contacts table exactly as scoped by plan Part B: (owner_id, contact_id).
class Contact(Base):
    """Represent one address-book edge owned by a signed-in account.

    The server stores only who-knows-whom metadata. It never stores a
    message body, a score, or any key material. Spec §11 forbids keeping
    this list in localStorage; this table is the replacement.
    """

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "contacts"

    # Identify each contact edge with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify the signed-in account that owns this address-book row.
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Identify the other account this owner saved (the contact's users.id).
    contact_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # Record when the owner saved this contact using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Enforce uniqueness and "no self-contact" at the database layer.
    __table_args__ = (
        # One row per (owner, contact) pair so Add is idempotent.
        UniqueConstraint("owner_id", "contact_id", name="uq_contacts_owner_contact"),
        # An account cannot add itself as a contact.
        CheckConstraint("owner_id != contact_id", name="ck_contacts_not_self"),
    )
