"""create conversations and messages tables

Revision ID: c4e8a2b91d07
Revises: 5fc1249c7176
Create Date: 2026-08-14 09:30:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "c4e8a2b91d07"
# Chain this migration directly after the refresh_tokens-table migration.
down_revision: str | Sequence[str] | None = "5fc1249c7176"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that creates conversations and messages.
def upgrade() -> None:
    """Create conversations and messages matching the Slice 4 ORM models.

    conversations holds only membership and a non-secret epoch counter.
    messages holds only ciphertext, nonce, and key_epoch — never plaintext
    or any private/symmetric key material.
    """

    op.create_table(
        "conversations",
        # Identify each conversation with a non-guessable random UUID.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Store the lexicographically-smaller participant UUID.
        sa.Column("user_a_id", sa.Uuid(as_uuid=True), nullable=False),
        # Store the lexicographically-larger participant UUID.
        sa.Column("user_b_id", sa.Uuid(as_uuid=True), nullable=False),
        # Store the non-secret epoch counter; Slice 4 reads it, later slices rotate it.
        sa.Column("current_epoch", sa.Integer(), server_default="0", nullable=False),
        # Record conversation creation time using the database server's clock.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Reject deleting an account that still has conversations (no silent orphaning).
        sa.ForeignKeyConstraint(["user_a_id"], ["users.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["user_b_id"], ["users.id"], ondelete="RESTRICT"),
        # One conversation row per unordered pair of accounts (plan Part B).
        sa.UniqueConstraint("user_a_id", "user_b_id", name="uq_conversations_user_pair"),
        # Canonicalize pair order so initiator identity cannot duplicate the row.
        sa.CheckConstraint("user_a_id < user_b_id", name="ck_conversations_user_order"),
    )
    # Speed up "find conversations this user is in" lookups from either side of the pair.
    op.create_index("ix_conversations_user_a_id", "conversations", ["user_a_id"], unique=False)
    op.create_index("ix_conversations_user_b_id", "conversations", ["user_b_id"], unique=False)

    op.create_table(
        "messages",
        # Identify each envelope with a non-guessable random UUID.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Scope every message row to exactly one conversation (§2).
        sa.Column("conversation_id", sa.Uuid(as_uuid=True), nullable=False),
        # Record which participant produced the envelope (routing metadata only).
        sa.Column("sender_id", sa.Uuid(as_uuid=True), nullable=False),
        # Store AEAD ciphertext bytes (including the Poly1305 tag); never plaintext.
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        # Store the public nonce required for client-side authenticated decryption.
        sa.Column("nonce", sa.LargeBinary(), nullable=False),
        # Store the non-secret epoch id used to derive the message key on the clients.
        sa.Column("key_epoch", sa.Integer(), nullable=False),
        # Record insertion time using the database server's clock.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Drop stored envelopes if the parent conversation is ever deleted.
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        # Reject deleting an account that still has sent envelopes.
        sa.ForeignKeyConstraint(["sender_id"], ["users.id"], ondelete="RESTRICT"),
    )
    # Index (conversation_id, created_at DESC) so history queries never scan the whole table.
    op.create_index(
        "ix_messages_conversation_id_created_at",
        "messages",
        ["conversation_id", sa.text("created_at DESC")],
        unique=False,
    )


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the messages and conversations tables and their indexes."""

    # Drop the dependent messages table first so the conversations FK is not violated.
    op.drop_index("ix_messages_conversation_id_created_at", table_name="messages")
    op.drop_table("messages")
    # Drop conversation membership indexes before the table itself.
    op.drop_index("ix_conversations_user_b_id", table_name="conversations")
    op.drop_index("ix_conversations_user_a_id", table_name="conversations")
    # Drop the conversations table, completing the rollback.
    op.drop_table("conversations")
