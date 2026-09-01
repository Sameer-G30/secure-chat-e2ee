"""Expose authenticated username search, replacing the legacy full-table-scan approach."""

# Import Annotated for dependency metadata.
from typing import Annotated

# Import FastAPI's routing, dependency, and query-parameter primitives.
from fastapi import APIRouter, Depends, Query

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model the auth dependency returns.
from app.models.user import User

# Import the validated response shape and bounds for this router's endpoint.
from app.schemas.users import (
    MAX_USER_SEARCH_RESULTS,
    MIN_USER_SEARCH_QUERY_LENGTH,
    UserSearchResponse,
)

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user

# Import the search helper so the router stays thin.
from app.services.users import search_users

# Group user-discovery REST under one versionable tag; paths are absolute.
router = APIRouter(tags=["users"])


# Search for accounts by a case-insensitive username prefix.
@router.get("/users/search", response_model=UserSearchResponse)
async def search_users_by_prefix(
    # Require a valid access token; this replaces the legacy app's approach of
    # downloading every user row to an unauthenticated browser tab.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
    # Require a minimum query length so a one-character query cannot return a
    # large fraction of the user base (matches the legacy React app's own
    # `useContacts.js` two-character minimum).
    q: Annotated[str, Query(min_length=MIN_USER_SEARCH_QUERY_LENGTH, max_length=32)],
    # Allow the caller to ask for fewer results; never more than the server cap.
    limit: Annotated[int, Query(ge=1, le=MAX_USER_SEARCH_RESULTS)] = MAX_USER_SEARCH_RESULTS,
) -> UserSearchResponse:
    """Return up to `limit` usernames starting with `q`, excluding the caller.

    Never returns email, password hash, or public key — see
    app/schemas/users.py's UserSearchResult.
    """

    return await search_users(db, q, current_user.id, limit=limit)
