"""Provide the shared FastAPI dependency that authenticates protected routes."""

# Import Annotated to describe dependency-injected parameter metadata.
from typing import Annotated

# Import UUID to parse the token subject claim into a typed identifier.
from uuid import UUID

# Import FastAPI's dependency injection and HTTP-error primitives.
from fastapi import Depends, HTTPException, status

# Import FastAPI's bearer-scheme helpers so Swagger UI shows the Authorization header.
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Import SQLAlchemy's async session type used to load the authenticated account.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model this dependency loads and returns.
from app.models.user import User

# Import the access-token verifier and its dedicated error type.
from app.security.tokens import ACCESS_TOKEN_TYPE, TokenError, decode_token

# Declare one shared bearer-token security scheme for every protected router.
_bearer_scheme = HTTPBearer(
    # Reject requests with no Authorization header before the endpoint body runs.
    auto_error=True,
    # Describe the expected header for generated API documentation.
    description="Short-lived JWT access token issued by POST /auth/login",
)


# Resolve the authenticated account from a raw access-token string.
async def get_user_from_access_token(raw_token: str, db: AsyncSession) -> User:
    """Return the User row identified by a verified access token's subject claim.

    Used by both HTTP Bearer auth and the WebSocket query-string handshake so
    the two paths cannot drift onto different verification rules.
    """

    try:
        # Verify signature, expiry, and that this is specifically an access token.
        payload = decode_token(raw_token, expected_type=ACCESS_TOKEN_TYPE)
    except TokenError as exc:
        # Never distinguish "expired" from "malformed" to callers; both are just unauthenticated.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="invalid or expired access token"
        ) from exc

    try:
        # Parse the token's declared subject into the identifier type the database expects.
        user_id = UUID(str(payload.get("sub")))
    except ValueError as exc:
        # A syntactically valid JWT with a malformed subject is still not usable.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="access token subject is invalid"
        ) from exc

    # Load the account fresh on every request rather than trusting stale token claims.
    user = await db.get(User, user_id)
    if user is None:
        # The account may have been deleted after the token was issued.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="account for this token no longer exists"
        )
    return user


# Resolve the authenticated account from a validated bearer access token.
async def get_current_user(
    # Extract and require a well-formed "Authorization: Bearer <token>" header.
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    # Inject a request-scoped database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Return the User row identified by the request's Authorization bearer token."""

    # Delegate to the shared verifier so HTTP and WebSocket auth stay identical.
    return await get_user_from_access_token(credentials.credentials, db)
