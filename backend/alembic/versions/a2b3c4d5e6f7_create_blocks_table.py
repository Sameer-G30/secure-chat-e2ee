"""create blocks table

Revision ID: a2b3c4d5e6f7
Revises: f1c2a3b4d5e6
Create Date: 2026-09-01 09:05:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "a2b3c4d5e6f7"
# Chain this migration directly after the message-edit-columns migration.
down_revision: str | Sequence[str] | None = "f1c2a3b4d5e6"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that creates server-enforced blocking.
def upgrade() -> None:
    """Create blocks matching the pre-deployment-review ORM model.

    Unlike the legacy app's localStorage-only "block," this table is the
    server-side truth the relay layer actually enforces.
    """

    op.create_table(
        "blocks",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("blocker_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("blocked_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["blocker_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["blocked_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("blocker_id", "blocked_id", name="uq_blocks_blocker_blocked"),
        sa.CheckConstraint("blocker_id != blocked_id", name="ck_blocks_not_self"),
    )
    op.create_index("ix_blocks_blocker_id", "blocks", ["blocker_id"], unique=False)
    op.create_index("ix_blocks_blocked_id", "blocks", ["blocked_id"], unique=False)


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the blocks table and its lookup indexes."""

    op.drop_index("ix_blocks_blocked_id", table_name="blocks")
    op.drop_index("ix_blocks_blocker_id", table_name="blocks")
    op.drop_table("blocks")
