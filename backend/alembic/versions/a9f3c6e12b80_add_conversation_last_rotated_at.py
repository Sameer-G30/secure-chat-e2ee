"""add conversations.last_rotated_at for epoch rotation scheduling

Revision ID: a9f3c6e12b80
Revises: e7b4d2a81c09
Create Date: 2026-08-25 14:40:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "a9f3c6e12b80"
# Chain this migration directly after the contacts-table migration.
down_revision: str | Sequence[str] | None = "e7b4d2a81c09"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that records when current_epoch last incremented.
def upgrade() -> None:
    """Add last_rotated_at so Slice 8 can enforce the 24h half of rotation.

    NULL means this conversation has never been bumped (treat as created_at
    for the wall-clock rule, and count every envelope for the N-message rule).
    This column is a timestamp, not a key.
    """

    # Add a nullable timestamptz; existing rows stay NULL until the first bump.
    op.add_column(
        "conversations",
        sa.Column("last_rotated_at", sa.DateTime(timezone=True), nullable=True),
    )


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop last_rotated_at; current_epoch itself stays on the table."""

    # Remove only the rotation timestamp; do not drop current_epoch.
    op.drop_column("conversations", "last_rotated_at")
