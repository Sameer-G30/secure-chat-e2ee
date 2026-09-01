"""Search accounts by username without ever downloading the whole users table.

Replaces the legacy React prototype's `useContacts.js` (`users` RTDB node
`.get()` — download every account row to the browser, substring-match in
JavaScript, which is both a scalability problem and a privacy leak: any signed-in
user could enumerate the entire user base). This is a server-side, index-backed,
prefix-only match instead.
"""

# Import UUID for the typed caller-identifier parameter.
from uuid import UUID

# Import SQLAlchemy's select and case-folding/limiting helpers.
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM model this service reads.
from app.models.user import User

# Import the response shape and bounds so routers do not rebuild the payload.
from app.schemas.users import MAX_USER_SEARCH_RESULTS, UserSearchResponse, UserSearchResult


# Search for accounts whose username starts with the caller's query, case-insensitively.
async def search_users(
    db: AsyncSession,
    query: str,
    requesting_user_id: UUID,
    *,
    limit: int = MAX_USER_SEARCH_RESULTS,
) -> UserSearchResponse:
    """Return up to `limit` (capped at MAX_USER_SEARCH_RESULTS) usernames starting with `query`.

    Case-insensitive prefix match only (never a substring/contains match, which
    would force a full-table scan even with the lower(username) index this
    query relies on). Excludes the caller's own account — you cannot "search"
    for yourself to add yourself as a contact. Never returns email, password
    hash, or public key: see app/schemas/users.py's UserSearchResult.
    """

    normalized = query.strip().lower()
    if not normalized:
        return UserSearchResponse(users=[])

    # Escape SQL LIKE wildcards in the caller's own query so "a_b" or "a%b" cannot
    # widen the match beyond a literal prefix.
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"{escaped}%"
    # Never trust a caller-supplied limit past the documented server-side cap.
    bounded_limit = max(1, min(limit, MAX_USER_SEARCH_RESULTS))

    rows = await db.scalars(
        select(User)
        .where(func.lower(User.username).like(pattern, escape="\\"))
        .where(User.id != requesting_user_id)
        .order_by(User.username)
        .limit(bounded_limit)
    )
    return UserSearchResponse(users=[UserSearchResult(username=row.username) for row in rows])
