"""Validate block create/list payloads without carrying message bodies or scores."""

# Import datetime for the response's typed timestamp field.
from datetime import datetime

# Import UUID for typed account identifiers.
from uuid import UUID

# Import Pydantic's model base used by request and response payloads.
from pydantic import BaseModel

# Reuse the same URL-safe username type register and contacts already enforce.
from app.schemas.usernames import Username


# Validate the single field a block request must supply.
class BlockUserRequest(BaseModel):
    """Represent the client-submitted handle to block."""

    # Identify the other account by the same handle used for GET /keys/{username}.
    username: Username


# Describe one blocked account as returned to the authenticated blocker.
class BlockResponse(BaseModel):
    """Represent one block edge without any private key or message body."""

    # Identify the blocked account.
    id: UUID
    # Identify the blocked account with the handle the UI displays.
    username: str
    # Record when the block was created.
    created_at: datetime


# Describe the caller's full block list.
class BlockListResponse(BaseModel):
    """Represent every account the authenticated caller has blocked."""

    # Carry the blocked accounts; empty when the caller has not blocked anyone.
    blocks: list[BlockResponse]
