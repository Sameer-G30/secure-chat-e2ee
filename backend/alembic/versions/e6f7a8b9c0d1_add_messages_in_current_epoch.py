"""add O(1) messages_in_current_epoch counter on conversations

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-09-01 16:40:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import SQLAlchemy column types used in the upgrade.
import sqlalchemy as sa

# Import Alembic's schema-operation helpers.
from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "e6f7a8b9c0d1"
# Chain after the username-lower index migration.
down_revision: str | Sequence[str] | None = "d5e6f7a8b9c0"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that replaces per-persist COUNT(*) for epoch rotation.
def upgrade() -> None:
    """Add messages_in_current_epoch and backfill from existing message rows."""

    op.add_column(
        "conversations",
        sa.Column(
            "messages_in_current_epoch",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    # Backfill so a deployment with existing history does not under-count.
    op.execute(
        """
        UPDATE conversations
        SET messages_in_current_epoch = (
            SELECT COUNT(*)
            FROM messages
            WHERE messages.conversation_id = conversations.id
              AND (
                  conversations.last_rotated_at IS NULL
                  OR messages.created_at > conversations.last_rotated_at
              )
        )
        """
    )


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the O(1) epoch message counter."""

    op.drop_column("conversations", "messages_in_current_epoch")
