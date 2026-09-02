"""Define opaque encrypted file bytes the server may store for one conversation.

The server never decrypts these bytes. The file key travels inside a normal
E2EE chat envelope, not in this table. Nonce is public (same as message nonces).
"""

# Import datetime for typed created_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations.
from uuid import UUID

# Import binary, timestamp, foreign-key, integer, and UUID column types.
from sqlalchemy import DateTime, ForeignKey, Index, Integer, LargeBinary, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


class EncryptedBlob(Base):
    """Represent one client-encrypted image blob scoped to a conversation."""

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "encrypted_blobs"

    # Identify the blob with a client-chosen UUID so file associated data can bind it
    # before upload (the server must not assign an id after the file is already encrypted).
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    # Identify which conversation this blob belongs to; every read filters on this.
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Identify which member uploaded the opaque bytes (metadata, not a crypto key).
    uploader_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    # Store the AEAD ciphertext (including the Poly1305 tag); never file pixels in the clear.
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Store the public nonce required for authenticated decryption on the client.
    nonce: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Record the stored byte length so list/quota checks do not load the blob.
    byte_length: Mapped[int] = mapped_column(Integer, nullable=False)
    # Record insertion time using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Serve conversation-scoped blob lookups without a table scan.
    __table_args__ = (
        Index("ix_encrypted_blobs_conversation_id", "conversation_id"),
    )
