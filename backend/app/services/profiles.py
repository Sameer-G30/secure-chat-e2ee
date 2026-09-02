"""Load and update public profile metadata without touching message envelopes."""

# Import FastAPI's HTTP-error primitives so unknown handles stay 404.
from fastapi import HTTPException, status

# Import SQLAlchemy query helpers used for username lookups.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM model this service reads and writes.
from app.models.user import User

# Import the response shapes so routers do not rebuild profile payloads.
from app.schemas.profiles import MeProfileResponse, PatchMeRequest, PublicProfileResponse

# Shared detail when the named account does not exist.
_USER_NOT_FOUND_DETAIL = "user not found"


# Build the authenticated caller's own profile payload.
def serialize_me(user: User) -> MeProfileResponse:
    """Return username, email, display name, bio, and whether an avatar exists."""

    return MeProfileResponse(
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        bio=user.bio,
        has_avatar=user.avatar_bytes is not None,
    )


# Build another account's public profile payload (no email).
def serialize_public_profile(user: User) -> PublicProfileResponse:
    """Return username, display name, bio, and whether an avatar exists."""

    return PublicProfileResponse(
        username=user.username,
        display_name=user.display_name,
        bio=user.bio,
        has_avatar=user.avatar_bytes is not None,
    )


# Load a public profile by handle.
async def get_public_profile(db: AsyncSession, username: str) -> PublicProfileResponse:
    """Return one account's public profile, or 404 if the handle is unknown."""

    user = await db.scalar(select(User).where(User.username == username))
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_USER_NOT_FOUND_DETAIL)
    return serialize_public_profile(user)


# Load an account row by handle for avatar serving.
async def get_user_by_username(db: AsyncSession, username: str) -> User:
    """Return the User row for a handle, or 404 if the handle is unknown."""

    user = await db.scalar(select(User).where(User.username == username))
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_USER_NOT_FOUND_DETAIL)
    return user


# Apply a PATCH /users/me update for fields the caller actually sent.
async def patch_me(db: AsyncSession, user: User, payload: PatchMeRequest) -> MeProfileResponse:
    """Update display_name and/or bio when those fields are present on the request."""

    if "display_name" in payload.model_fields_set:
        # Empty string clears the stored display name.
        trimmed = (payload.display_name or "").strip()
        user.display_name = trimmed or None
    if "bio" in payload.model_fields_set:
        # Empty string clears the stored bio.
        trimmed = (payload.bio or "").strip()
        user.bio = trimmed or None
    await db.commit()
    await db.refresh(user)
    return serialize_me(user)


# Replace the caller's public avatar bytes and media type.
async def replace_avatar(
    db: AsyncSession,
    user: User,
    *,
    image_bytes: bytes,
    media_type: str,
) -> MeProfileResponse:
    """Store public avatar bytes; this is not a chat envelope and is not E2EE."""

    user.avatar_bytes = image_bytes
    user.avatar_media_type = media_type
    await db.commit()
    await db.refresh(user)
    return serialize_me(user)
