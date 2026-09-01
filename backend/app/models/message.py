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
    # Version the associated-data format bound into this envelope's AEAD tag.
    #
    # 1 = the original format (`['secure-chat-envelope-v1', conversation_id,
    # sender_id, key_epoch]`), which binds no message identity. 2 = the format
    # added for message editing (`['secure-chat-envelope-v2', conversation_id,
    # sender_id, key_epoch, client_message_id, revision]`). This is metadata
    # about how to *reconstruct* the associated data on decrypt; the server
    # still never sees a key or plaintext. Existing v1 history is never
    # rewritten, so it keeps decrypting exactly as before.
    ad_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    # Carry the client-generated identity this envelope's AD is bound to (v2 only).
    #
    # Must be chosen by the sender's device *before* encryption, because the AD is
    # authenticated inside the ciphertext tag — the server cannot assign an id
    # after the fact the way it does for the primary key `id` column above. NULL
    # for v1 envelopes, which have no message identity at all.
    client_message_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    # Count edits to one v2 message: 0 for the original send, 1 for the first
    # edit, and so on. Bound into the v2 AD so a server cannot replay an older
    # revision's ciphertext as if it were still current (the edit-rollback gap
    # a naive edit feature would otherwise introduce).
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # Record the most recent edit time; NULL means this envelope was never edited.
    # A timestamp only — never the edited plaintext, which the server cannot read.
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Serve the §2 requirement that message queries are scoped by conversation_id.
    __table_args__ = (
        # Index (conversation_id, created_at DESC) per plan Part B.
        Index(
            "ix_messages_conversation_id_created_at",
            "conversation_id",
            created_at.desc(),
        ),
        # Look up "is this send an edit of an existing v2 message" by
        # (conversation_id, client_message_id) without a conversation-wide scan.
        # Not a unique index: SQLite/Postgres partial-unique syntax diverges, and
        # `relay_envelope` already enforces one-row-per-client_message_id at the
        # application layer (see app/services/relay.py) before ever reaching here.
        Index(
            "ix_messages_conversation_id_client_message_id",
            "conversation_id",
            "client_message_id",
        ),
    )
