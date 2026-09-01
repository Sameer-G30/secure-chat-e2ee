"""create message_hides table

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-09-01 09:15:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "c4d5e6f7a8b9"
# Chain this migration directly after the reports-table migration.
down_revision: str | Sequence[str] | None = "b3c4d5e6f7a8"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that creates the per-owner "delete for me" marker.
def upgrade() -> None:
    """Create message_hides matching the pre-deployment-review ORM model.

    Scoped per-owner so hiding a message from your own history never affects
    the peer's copy of the same envelope (unlike the legacy app's buggy
    delete-for-me, which wrote a shared `deleted` flag onto the message row).
    """

    op.create_table(
        "message_hides",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("owner_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("message_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("owner_id", "message_id", name="uq_message_hides_owner_message"),
    )
    op.create_index("ix_message_hides_owner_id", "message_hides", ["owner_id"], unique=False)
    op.create_index("ix_message_hides_message_id", "message_hides", ["message_id"], unique=False)


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the message_hides table and its lookup indexes."""

    op.drop_index("ix_message_hides_message_id", table_name="message_hides")
    op.drop_index("ix_message_hides_owner_id", table_name="message_hides")
    op.drop_table("message_hides")
