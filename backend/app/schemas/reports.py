"""Validate metadata-only abuse-report payloads. Never a message body."""

# Import datetime for the response's typed timestamp field.
from datetime import datetime

# Import UUID for the response's typed identifier field.
from uuid import UUID

# Import Pydantic's model base and field-length helper.
from pydantic import BaseModel, Field

# Reuse the same URL-safe username type register and contacts already enforce.
from app.schemas.usernames import Username

# Cap the free-text reason so a report cannot smuggle unbounded data (matches
# app/models/report.py's MAX_REPORT_REASON_LENGTH).
MAX_REPORT_REASON_LENGTH = 500


# Validate the fields a report request must supply.
class ReportUserRequest(BaseModel):
    """Represent a metadata-only abuse report the caller files against another account.

    There is deliberately no field for message content or a message id: the
    server cannot read message ciphertext, so it cannot verify — or even
    accept — reported message text without breaking the E2EE trust boundary.
    """

    # Identify the reported account by the same handle used for GET /keys/{username}.
    username: Username
    # Carry the reporter's free-text reason, bounded so this cannot smuggle unbounded data.
    reason: str = Field(min_length=1, max_length=MAX_REPORT_REASON_LENGTH)


# Describe the report record returned to the filer as confirmation.
class ReportResponse(BaseModel):
    """Represent one filed report without any private key or message body."""

    # Identify the persisted report row.
    id: UUID
    # Record when the report was filed.
    created_at: datetime
