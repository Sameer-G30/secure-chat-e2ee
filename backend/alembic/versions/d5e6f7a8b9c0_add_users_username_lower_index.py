"""add functional index on lower(users.username) for GET /users/search

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-09-01 09:20:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "d5e6f7a8b9c0"
# Chain this migration directly after the message_hides-table migration.
down_revision: str | Sequence[str] | None = "c4d5e6f7a8b9"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that speeds up the new case-insensitive username search.
def upgrade() -> None:
    """Create a functional index on lower(username) for GET /users/search.

    Replaces the legacy app's approach (`users` RTDB node `.get()` — download
    every account row to the browser, substring-match in JavaScript) with a
    server-side, index-backed prefix match.
    """

    op.execute("CREATE INDEX ix_users_username_lower ON users (lower(username))")


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the functional username index."""

    op.execute("DROP INDEX ix_users_username_lower")
