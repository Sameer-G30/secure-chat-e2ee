"""Define the ciphertext-only message envelope the server may store and relay."""

# Import datetime for typed created_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import binary, timestamp, foreign-key, index, integer, and UUID column types.
from sqlalchemy import DateTime, ForeignKey, Index, Integer, LargeBinary, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


# Map the messages table exactly as scoped by spec §5 and plan Part B.
class Message(Base):
    """Represent one stored ciphertext envelope.

    Only ciphertext, nonce, and a non-secret key_epoch live here. There is no
    plaintext column, no body column, and no key material. Every query against
    this table must be scoped by conversation_id (§2, §11).
    """

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "messages"

    # Identify each envelope with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify which conversation this envelope belongs to; all reads filter on this.
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Identify which participant produced the envelope (metadata, not a crypto key).
    sender_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    # Store the AEAD ciphertext (including the Poly1305 tag); never plaintext.
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Store the public nonce required for authenticated decryption on the client.
    nonce: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Store the non-secret epoch id the clients used to derive the message key.
    key_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    # Record insertion time using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Serve the §2 requirement that message queries are scoped by conversation_id.
    __table_args__ = (
        # Index (conversation_id, created_at DESC) per plan Part B.
        Index(
            "ix_messages_conversation_id_created_at",
            "conversation_id",
            created_at.desc(),
        ),
    )
