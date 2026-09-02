"""Expose authenticated username search and public profile metadata."""

# Import Annotated for dependency metadata.
from typing import Annotated

# Import FastAPI's routing, dependency, file-upload, query, and response primitives.
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model the auth dependency returns.
from app.models.user import User

# Import the validated response shape and bounds for this router's search endpoint.
from app.schemas.profiles import (
    ALLOWED_AVATAR_MEDIA_TYPES,
    MAX_AVATAR_BYTES,
    MeProfileResponse,
    PatchMeRequest,
    PublicProfileResponse,
)
from app.schemas.users import (
    MAX_USER_SEARCH_RESULTS,
    MIN_USER_SEARCH_QUERY_LENGTH,
    UserSearchResponse,
)

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user

# Import profile load/update helpers so the router stays thin.
from app.services.profiles import (
    get_public_profile,
    get_user_by_username,
    patch_me,
    replace_avatar,
    serialize_me,
)
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


# Return the signed-in account's editable public profile.
@router.get("/users/me", response_model=MeProfileResponse)
async def fetch_me(
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
) -> MeProfileResponse:
    """Return username, email, display name, bio, and whether an avatar exists."""

    return serialize_me(current_user)


# Update the signed-in account's display name and/or bio.
@router.patch("/users/me", response_model=MeProfileResponse)
async def update_me(
    # Accept the optional public fields the owner wants to change.
    payload: PatchMeRequest,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MeProfileResponse:
    """Patch display_name and bio. Omitted fields are left unchanged."""

    return await patch_me(db, current_user, payload)


# Replace the signed-in account's public avatar image.
@router.post("/users/me/avatar", response_model=MeProfileResponse)
async def upload_my_avatar(
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
    # Accept one JPEG/PNG/WebP file whose bytes are public profile metadata.
    file: Annotated[UploadFile, File()],
) -> MeProfileResponse:
    """Store a public avatar. This is not a chat envelope and is not E2EE."""

    media_type = (file.content_type or "").split(";")[0].strip().lower()
    if media_type not in ALLOWED_AVATAR_MEDIA_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="avatar must be image/jpeg, image/png, or image/webp",
        )
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="avatar file is empty")
    if len(image_bytes) > MAX_AVATAR_BYTES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="avatar exceeds 200KB")
    return await replace_avatar(
        db, current_user, image_bytes=image_bytes, media_type=media_type
    )


# Return another account's public profile (no email, hash, or key).
@router.get("/users/{username}/profile", response_model=PublicProfileResponse)
async def fetch_public_profile(
    # Identify which handle to look up.
    username: str,
    # Require a valid access token so this is not an unauthenticated user oracle.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PublicProfileResponse:
    """Return display name, bio, and avatar flag for a named account."""

    # current_user is required so only signed-in members can look up profiles.
    _ = current_user
    return await get_public_profile(db, username)


# Return another account's public avatar bytes, or 404 when none is stored.
@router.get("/users/{username}/avatar")
async def fetch_public_avatar(
    # Identify which handle's avatar to return.
    username: str,
    # Require a valid access token so avatars are not a public unauthenticated dump.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Return raw image bytes with the stored media type. Auth is required."""

    _ = current_user
    user = await get_user_by_username(db, username)
    if user.avatar_bytes is None or user.avatar_media_type is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="avatar not found")
    return Response(content=user.avatar_bytes, media_type=user.avatar_media_type)
