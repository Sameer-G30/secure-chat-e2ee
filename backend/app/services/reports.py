"""File metadata-only abuse reports. Never accepts or stores message content."""

# Import FastAPI's HTTP-error primitives so callers can return stable status codes.
from fastapi import HTTPException, status

# Import SQLAlchemy's select helper for the reported-user lookup.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM models this service reads and writes.
from app.models.report import Report
from app.models.user import User

# Import the response shape so routers do not rebuild the report payload.
from app.schemas.reports import ReportResponse

# Shared detail when the named account does not exist.
_USER_NOT_FOUND_DETAIL = "user not found"
# Shared detail when the caller tries to report their own account.
_SELF_REPORT_DETAIL = "cannot report yourself"


# File one report. Always creates a new row (unlike contacts/blocks, repeated
# reports about ongoing abuse are meaningful signal, not a duplicate to collapse).
async def file_report(
    db: AsyncSession, reporter: User, username: str, reason: str
) -> ReportResponse:
    """Insert one (reporter_id, reported_id, reason) row. Metadata only.

    There is no message-content parameter here by design: the server cannot
    read message ciphertext, so it structurally cannot verify — or even
    accept — reported message text without breaking the E2EE trust boundary
    documented in the architecture diagram.
    """

    if username == reporter.username:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=_SELF_REPORT_DETAIL)

    reported_user = await db.scalar(select(User).where(User.username == username))
    if reported_user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_USER_NOT_FOUND_DETAIL)

    report = Report(reporter_id=reporter.id, reported_id=reported_user.id, reason=reason)
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return ReportResponse(id=report.id, created_at=report.created_at)
