"""Expose authenticated REST to file a metadata-only abuse report."""

# Import Annotated for dependency metadata.
from typing import Annotated

# Import FastAPI's routing and dependency primitives.
from fastapi import APIRouter, Depends, status

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model the auth dependency returns.
from app.models.user import User

# Import the validated request and response shapes for this router's endpoint.
from app.schemas.reports import ReportResponse, ReportUserRequest

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user

# Import the report-filing helper so the router stays thin.
from app.services.reports import file_report

# Group report REST under one versionable tag; paths are absolute.
router = APIRouter(tags=["reports"])


# File a metadata-only abuse report against a named account.
@router.post("/reports", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def report_user(
    payload: ReportUserRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ReportResponse:
    """Record who reported whom and why. Never accepts message content.

    The server has no key to read message ciphertext, so it structurally
    cannot verify — or even accept — reported message text without breaking
    the E2EE trust boundary in the architecture diagram.
    """

    return await file_report(db, current_user, payload.username, payload.reason)
