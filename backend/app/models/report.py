"""Define the metadata-only abuse report row.

The legacy React prototype's report feature could only ever carry metadata too
(`reporterUid`, `reportedUid`, `reason`) — Firebase Storage/Firestore were never
wired to attach the reported message. In this project that limitation is a hard
requirement, not an omission: the server has no key to read message ciphertext,
so it structurally cannot attach message content to a report without breaking
the E2EE trust boundary in the architecture diagram. This table never gains a
message-content column.
"""

# Import datetime for typed created_at annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import constraint helpers, timestamps, foreign keys, and the portable UUID type.
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base

# Cap the free-text reason so a report cannot be used to smuggle unbounded data.
MAX_REPORT_REASON_LENGTH = 500


class Report(Base):
    """Represent one metadata-only abuse report filed by one account against another."""

    __tablename__ = "reports"

    # Identify each report with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify the account that filed the report.
    reporter_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Identify the account being reported.
    reported_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Store the reporter's free-text reason; never message ciphertext or plaintext.
    reason: Mapped[str] = mapped_column(String(MAX_REPORT_REASON_LENGTH), nullable=False)
    # Record when the report was filed using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # An account cannot report itself.
        CheckConstraint("reporter_id != reported_id", name="ck_reports_not_self"),
    )
