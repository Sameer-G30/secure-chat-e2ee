"""add messages.ad_version/client_message_id/revision/edited_at for safe message editing

Revision ID: f1c2a3b4d5e6
Revises: a9f3c6e12b80
Create Date: 2026-09-01 09:00:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "f1c2a3b4d5e6"
# Chain this migration directly after the epoch-rotation-timestamp migration.
down_revision: str | Sequence[str] | None = "a9f3c6e12b80"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that lets a v2 envelope carry a client-chosen message
# identity and a revision count, enabling safe message editing.
def upgrade() -> None:
    """Add ad_version/client_message_id/revision/edited_at to messages.

    ad_version=1 (the default) matches every existing row's associated-data
    format exactly: `['secure-chat-envelope-v1', conversation_id, sender_id,
    key_epoch]`. No existing ciphertext is touched or re-encrypted. New sends
    from an updated client use ad_version=2, which additionally binds
    client_message_id and revision into the AEAD tag, closing the
    edit-rollback gap a naive edit feature would otherwise introduce (the
    server could otherwise serve a pre-edit ciphertext and the recipient
    would accept it as authentic). None of these four columns are plaintext
    or key material — see backend/tests/test_ciphertext_boundary.py.
    """

    op.add_column(
        "messages",
        sa.Column("ad_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "messages",
        sa.Column("client_message_id", sa.Uuid(as_uuid=True), nullable=True),
    )
    op.add_column(
        "messages",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "messages",
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Look up "is this send an edit of an existing v2 message" by
    # (conversation_id, client_message_id) without a conversation-wide scan.
    op.create_index(
        "ix_messages_conversation_id_client_message_id",
        "messages",
        ["conversation_id", "client_message_id"],
        unique=False,
    )


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the edit-support columns and their lookup index."""

    op.drop_index("ix_messages_conversation_id_client_message_id", table_name="messages")
    op.drop_column("messages", "edited_at")
    op.drop_column("messages", "revision")
    op.drop_column("messages", "client_message_id")
    op.drop_column("messages", "ad_version")
