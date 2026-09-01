"""Expose authenticated REST to block/unblock/list.

Replaces the legacy React prototype's localStorage-only, never-enforced block feature.
"""

# Import Annotated for dependency metadata.
from typing import Annotated

# Import FastAPI's routing, dependency, and status primitives.
from fastapi import APIRouter, Depends, status

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model the auth dependency returns.
from app.models.user import User

# Import the validated request and response shapes for this router's endpoints.
from app.schemas.blocks import BlockListResponse, BlockResponse, BlockUserRequest

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user

# Import block create/remove/list helpers so the router stays thin.
from app.services.blocks import block_user_for_owner, list_blocks_for_owner, unblock_user_for_owner

# Group block REST under one versionable tag; paths are absolute.
router = APIRouter(tags=["blocks"])


# Return the authenticated caller's block list.
@router.get("/blocks", response_model=BlockListResponse)
async def list_blocks(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BlockListResponse:
    """Return every account the caller has blocked.

    Unlike the legacy app's `localStorage`-only block list, this is
    server-side and is what the relay layer actually enforces (see
    app/services/relay.py).
    """

    return await list_blocks_for_owner(db, current_user)


# Block a named account.
@router.post("/blocks", response_model=BlockResponse)
async def block_user(
    payload: BlockUserRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BlockResponse:
    """Block by username. Idempotent when the block already exists."""

    return await block_user_for_owner(db, current_user, payload.username)


# Unblock a named account.
@router.delete("/blocks/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def unblock_user(
    username: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove a block by username. Never errors when the account was not blocked."""

    await unblock_user_for_owner(db, current_user, username)
