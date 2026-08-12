"""create users table

Revision ID: ec9138d5ae56
Revises:
Create Date: 2026-08-12 11:01:09.736211

"""

# Import Sequence/Union for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "ec9138d5ae56"
# Mark this as the first migration in the project.
down_revision: str | Sequence[str] | None = None
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for the first migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that creates the users table.
def upgrade() -> None:
    """Create the users table matching app.models.user.User."""

    # Create the table with exactly the columns the Slice 2 User model defines.
    op.create_table(
        "users",
        # Identify each account with a non-guessable random UUID.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Store the unique handle used for login and future key lookup.
        sa.Column("username", sa.String(), nullable=False),
        # Store the unique account email used for registration and recovery.
        sa.Column("email", sa.String(), nullable=False),
        # Store only the Argon2id hash output, never the plaintext password.
        sa.Column("password_hash", sa.String(), nullable=False),
        # Store the base64 X25519 public key; nullable until Slice 3's key upload lands.
        sa.Column("public_key", sa.String(), nullable=True),
        # Record account creation time using the database server's clock.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # Enforce username uniqueness at the database layer, not only in application code.
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)
    # Enforce email uniqueness at the database layer, not only in application code.
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the users table and its indexes."""

    # Drop indexes before the table so the operation is safely reversible.
    op.drop_index(op.f("ix_users_email"), table_name="users")
    # Drop the username uniqueness index.
    op.drop_index(op.f("ix_users_username"), table_name="users")
    # Drop the table itself, completing the rollback.
    op.drop_table("users")
