"""Validate public profile payloads that are never message bodies or keys."""

# Import Pydantic's model base and field bounds.
from pydantic import BaseModel, Field

# Cap a public display name so it fits the users.display_name column.
MAX_DISPLAY_NAME_LENGTH = 64
# Cap a public bio so it fits the users.bio column.
MAX_BIO_LENGTH = 280
# Cap uploaded avatar bytes (JPEG/PNG/WebP only).
MAX_AVATAR_BYTES = 200 * 1024
# Accept only these public image media types for avatars.
ALLOWED_AVATAR_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


# Describe the authenticated caller's own public profile plus email.
class MeProfileResponse(BaseModel):
    """Represent the signed-in account's editable public profile."""

    # Identify the account with the login handle.
    username: str
    # Carry the account email the owner already knows; never returned on public profiles.
    email: str
    # Carry the optional public display name.
    display_name: str | None = None
    # Carry the optional public bio.
    bio: str | None = None
    # Signal whether GET /users/{username}/avatar will return image bytes.
    has_avatar: bool = False


# Validate the fields PATCH /users/me may change.
class PatchMeRequest(BaseModel):
    """Represent an optional display-name and bio update.

    Omitted fields are left unchanged. Empty strings clear the stored value.
    """

    # Replace or clear the public display name when this field is present.
    display_name: str | None = Field(default=None, max_length=MAX_DISPLAY_NAME_LENGTH)
    # Replace or clear the public bio when this field is present.
    bio: str | None = Field(default=None, max_length=MAX_BIO_LENGTH)


# Describe another account's public profile (no email, hash, or key).
class PublicProfileResponse(BaseModel):
    """Represent one account's public profile metadata."""

    # Identify the account with the login handle.
    username: str
    # Carry the optional public display name.
    display_name: str | None = None
    # Carry the optional public bio.
    bio: str | None = None
    # Signal whether this account has uploaded an avatar.
    has_avatar: bool = False
