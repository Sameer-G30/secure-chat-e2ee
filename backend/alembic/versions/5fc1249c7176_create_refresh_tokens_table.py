"""create refresh_tokens table

Revision ID: 5fc1249c7176
Revises: ec9138d5ae56
Create Date: 2026-08-13 12:00:00.000000

"""

# Import Sequence/Union for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "5fc1249c7176"
# Chain this migration directly after the users-table migration.
down_revision: str | Sequence[str] | None = "ec9138d5ae56"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that creates the refresh_tokens table.
def upgrade() -> None:
    """Create refresh_tokens matching app.models.refresh_token.RefreshToken.

    Only a SHA-256 hash of each issued JWT is stored here (see
    app.security.tokens.hash_refresh_token); the raw token is never
    persisted, matching the spec's ban on server-side key/secret storage.
    """

    op.create_table(
        "refresh_tokens",
        # Identify each issued token with a non-guessable random UUID.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Reference the account this refresh token authenticates future rotations for.
        sa.Column("user_id", sa.Uuid(as_uuid=True), nullable=False),
        # Store only the SHA-256 hash of the issued JWT, never the raw token.
        sa.Column("token_hash", sa.String(), nullable=False),
        # Record issuance time for auditability, independent of the JWT's own iat claim.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Record the token's absolute expiry, mirrored from its JWT "exp" claim.
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        # Record when rotation, logout, or reuse-detection revoked this token.
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        # Cascade-delete a user's refresh tokens if the account itself is ever deleted.
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    # Enforce token-hash uniqueness at the database layer, not only in application code.
    op.create_index(
        op.f("ix_refresh_tokens_token_hash"), "refresh_tokens", ["token_hash"], unique=True
    )
    # Speed up "find this user's active tokens" queries used by reuse-detection revocation.
    op.create_index(op.f("ix_refresh_tokens_user_id"), "refresh_tokens", ["user_id"], unique=False)


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the refresh_tokens table and its indexes."""

    # Drop indexes before the table so the operation is safely reversible.
    op.drop_index(op.f("ix_refresh_tokens_user_id"), table_name="refresh_tokens")
    op.drop_index(op.f("ix_refresh_tokens_token_hash"), table_name="refresh_tokens")
    # Drop the table itself, completing the rollback.
    op.drop_table("refresh_tokens")
