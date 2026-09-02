"""add receipts, profile fields, and encrypted blobs

Revision ID: f8a9b0c1d2e3
Revises: e6f7a8b9c0d1
Create Date: 2026-09-02 23:59:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import SQLAlchemy column types used in the upgrade.
import sqlalchemy as sa

# Import Alembic's schema-operation helpers.
from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "f8a9b0c1d2e3"
# Chain after the messages_in_current_epoch counter migration.
down_revision: str | Sequence[str] | None = "e6f7a8b9c0d1"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply schema changes for receipts, public profile metadata, and opaque file blobs.
def upgrade() -> None:
    """Add conversation_reads, message_receipts, encrypted_blobs, and user profile columns.

    None of these columns is a message body or a key. Avatars are public metadata.
    Encrypted blobs are client-sealed bytes the server never opens.
    """

    # Add an optional public display name the sidebar and contact profile can show.
    op.add_column("users", sa.Column("display_name", sa.String(length=64), nullable=True))
    # Add an optional public bio shown on the contact profile panel.
    op.add_column("users", sa.Column("bio", sa.String(length=280), nullable=True))
    # Add optional public avatar bytes (JPEG/PNG/WebP); never a chat envelope.
    op.add_column("users", sa.Column("avatar_bytes", sa.LargeBinary(), nullable=True))
    # Add the avatar media type so GET /users/{username}/avatar can set Content-Type.
    op.add_column("users", sa.Column("avatar_media_type", sa.String(length=64), nullable=True))

    # Create per-member last-read cursors used only for unread badges.
    op.create_table(
        "conversation_reads",
        # Identify each cursor row with a non-guessable random UUID.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Identify the account whose unread badge this cursor drives.
        sa.Column("user_id", sa.Uuid(as_uuid=True), nullable=False),
        # Identify the conversation this cursor belongs to.
        sa.Column("conversation_id", sa.Uuid(as_uuid=True), nullable=False),
        # Record when this member last marked the conversation read.
        sa.Column(
            "last_read_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Optionally pin the newest envelope this member had focused.
        sa.Column("last_read_message_id", sa.Uuid(as_uuid=True), nullable=True),
        # Drop the cursor when the account is deleted.
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # Drop the cursor when the conversation is deleted.
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        # Forget a deleted envelope id rather than blocking message deletes.
        sa.ForeignKeyConstraint(
            ["last_read_message_id"], ["messages.id"], ondelete="SET NULL"
        ),
        # One cursor per (member, conversation).
        sa.UniqueConstraint(
            "user_id", "conversation_id", name="uq_conversation_reads_user_conversation"
        ),
    )
    # Speed up "load this member's cursor for one conversation" lookups.
    op.create_index(
        "ix_conversation_reads_user_id", "conversation_reads", ["user_id"], unique=False
    )
    # Speed up conversation-scoped cursor lookups.
    op.create_index(
        "ix_conversation_reads_conversation_id",
        "conversation_reads",
        ["conversation_id"],
        unique=False,
    )

    # Create delivered/read markers the peer's device reports for one envelope.
    op.create_table(
        "message_receipts",
        # Identify each receipt row with a non-guessable random UUID.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Identify the envelope these ticks describe.
        sa.Column("message_id", sa.Uuid(as_uuid=True), nullable=False),
        # Identify the member who received (not sent) the envelope.
        sa.Column("recipient_id", sa.Uuid(as_uuid=True), nullable=False),
        # Record when the recipient's device acknowledged the ciphertext.
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        # Record when the recipient focused the chat on that envelope.
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        # Cascade with hard delete-for-everyone so ticks cannot outlive the row.
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        # Drop the receipt when the recipient account is deleted.
        sa.ForeignKeyConstraint(["recipient_id"], ["users.id"], ondelete="CASCADE"),
        # One receipt row per (envelope, recipient).
        sa.UniqueConstraint(
            "message_id", "recipient_id", name="uq_message_receipts_message_recipient"
        ),
    )
    # Speed up "receipts for this envelope" lookups when serializing history.
    op.create_index(
        "ix_message_receipts_message_id", "message_receipts", ["message_id"], unique=False
    )
    # Speed up "receipts this recipient has written" lookups.
    op.create_index(
        "ix_message_receipts_recipient_id",
        "message_receipts",
        ["recipient_id"],
        unique=False,
    )

    # Create opaque encrypted file bytes scoped to one conversation.
    op.create_table(
        "encrypted_blobs",
        # Identify the blob with a client-chosen UUID bound into file associated data.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Identify which conversation this blob belongs to.
        sa.Column("conversation_id", sa.Uuid(as_uuid=True), nullable=False),
        # Identify which member uploaded the opaque bytes.
        sa.Column("uploader_id", sa.Uuid(as_uuid=True), nullable=False),
        # Store the AEAD ciphertext (including the Poly1305 tag).
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        # Store the public nonce required for authenticated decryption on the client.
        sa.Column("nonce", sa.LargeBinary(), nullable=False),
        # Record the stored byte length so list/quota checks do not load the blob.
        sa.Column("byte_length", sa.Integer(), nullable=False),
        # Record insertion time using the database server's clock.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Drop blobs when the conversation is deleted.
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        # Refuse deleting an account that still uploaded blobs (no silent orphaning).
        sa.ForeignKeyConstraint(["uploader_id"], ["users.id"], ondelete="RESTRICT"),
    )
    # Serve conversation-scoped blob lookups without a table scan.
    op.create_index(
        "ix_encrypted_blobs_conversation_id",
        "encrypted_blobs",
        ["conversation_id"],
        unique=False,
    )


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop receipts, blobs, and public profile columns added in this revision."""

    # Drop blob indexes before the table.
    op.drop_index("ix_encrypted_blobs_conversation_id", table_name="encrypted_blobs")
    # Drop the opaque file table.
    op.drop_table("encrypted_blobs")
    # Drop receipt indexes before the table.
    op.drop_index("ix_message_receipts_recipient_id", table_name="message_receipts")
    # Drop the envelope-side receipt index.
    op.drop_index("ix_message_receipts_message_id", table_name="message_receipts")
    # Drop the delivered/read table.
    op.drop_table("message_receipts")
    # Drop cursor indexes before the table.
    op.drop_index(
        "ix_conversation_reads_conversation_id", table_name="conversation_reads"
    )
    # Drop the member-side cursor index.
    op.drop_index("ix_conversation_reads_user_id", table_name="conversation_reads")
    # Drop the last-read cursor table.
    op.drop_table("conversation_reads")
    # Remove public profile columns from users.
    op.drop_column("users", "avatar_media_type")
    # Remove avatar bytes.
    op.drop_column("users", "avatar_bytes")
    # Remove bio.
    op.drop_column("users", "bio")
    # Remove display name.
    op.drop_column("users", "display_name")
