"""create reports table

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
Create Date: 2026-09-01 09:10:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "b3c4d5e6f7a8"
# Chain this migration directly after the blocks-table migration.
down_revision: str | Sequence[str] | None = "a2b3c4d5e6f7"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that creates metadata-only abuse reports.
def upgrade() -> None:
    """Create reports matching the pre-deployment-review ORM model.

    Metadata only (reporter, reported, a bounded free-text reason, and a
    timestamp) — the server has no key to read message ciphertext, so it
    structurally cannot attach reported message content here.
    """

    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("reporter_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("reported_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["reporter_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reported_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint("reporter_id != reported_id", name="ck_reports_not_self"),
    )
    op.create_index("ix_reports_reporter_id", "reports", ["reporter_id"], unique=False)
    op.create_index("ix_reports_reported_id", "reports", ["reported_id"], unique=False)


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the reports table and its lookup indexes."""

    op.drop_index("ix_reports_reported_id", table_name="reports")
    op.drop_index("ix_reports_reporter_id", table_name="reports")
    op.drop_table("reports")
