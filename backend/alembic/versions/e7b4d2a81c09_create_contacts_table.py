"""create contacts table

Revision ID: e7b4d2a81c09
Revises: c4e8a2b91d07
Create Date: 2026-08-23 21:30:00.000000

"""

# Import Sequence for Alembic's typed revision-identifier fields.
from collections.abc import Sequence

# Import Alembic's schema-operation helpers.
import sqlalchemy as sa

from alembic import op

# Identify this revision for Alembic's dependency graph.
revision: str = "e7b4d2a81c09"
# Chain this migration directly after the conversations-and-messages migration.
down_revision: str | Sequence[str] | None = "c4e8a2b91d07"
# No branch labels are needed for a single linear migration history.
branch_labels: str | Sequence[str] | None = None
# No cross-branch dependency is needed for this migration.
depends_on: str | Sequence[str] | None = None


# Apply the schema change that creates the server-side address book.
def upgrade() -> None:
    """Create contacts matching the Slice 7 ORM model.

    contacts holds only (owner_id, contact_id) metadata. Spec §11 forbids
    keeping this list in localStorage. There is no message-body column.
    """

    op.create_table(
        "contacts",
        # Identify each address-book edge with a non-guessable random UUID.
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
        # Identify the signed-in account that owns this row.
        sa.Column("owner_id", sa.Uuid(as_uuid=True), nullable=False),
        # Identify the other account this owner saved.
        sa.Column("contact_id", sa.Uuid(as_uuid=True), nullable=False),
        # Record when the owner saved this contact using the database server's clock.
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Drop address-book rows if either account is deleted.
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["contact_id"], ["users.id"], ondelete="CASCADE"),
        # One edge per (owner, contact) pair so Add is idempotent.
        sa.UniqueConstraint("owner_id", "contact_id", name="uq_contacts_owner_contact"),
        # An account cannot add itself as a contact.
        sa.CheckConstraint("owner_id != contact_id", name="ck_contacts_not_self"),
    )
    # Speed up "list my contacts" lookups by owner.
    op.create_index("ix_contacts_owner_id", "contacts", ["owner_id"], unique=False)


# Reverse the schema change for local rollback and migration testing.
def downgrade() -> None:
    """Drop the contacts table and its owner index."""

    # Drop the owner lookup index before the table itself.
    op.drop_index("ix_contacts_owner_id", table_name="contacts")
    # Drop the contacts table, completing the rollback.
    op.drop_table("contacts")
