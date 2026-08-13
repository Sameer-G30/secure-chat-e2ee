"""Expose authenticated public-key upload and lookup endpoints.

Both endpoints only ever handle X25519 *public* keys. The server has no
code path that receives, stores, or logs a private key: key generation and
sealing happen entirely client-side (frontend/src/crypto/keyExchange.ts and
keyVault.ts).
"""

# Import Annotated to describe dependency-injected parameter metadata.
from typing import Annotated

# Import FastAPI's routing, dependency, and HTTP-error primitives.
from fastapi import APIRouter, Depends, HTTPException, status

# Import SQLAlchemy's async session type and the scoped username lookup helper.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model this router reads and writes.
from app.models.user import User

# Import the validated request and response shapes for this router's endpoints.
from app.schemas.keys import PublicKeyResponse, PublicKeyUploadRequest

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user

# Group both key endpoints under one versionable prefix and tag.
router = APIRouter(prefix="/keys", tags=["keys"])


# Upload or replace the authenticated caller's own X25519 public key.
@router.post("/me", response_model=PublicKeyResponse)
async def upload_my_public_key(
    # Accept and validate the base64 public-key payload from the request body.
    payload: PublicKeyUploadRequest,
    # Require a valid access token; only the account owner may set their own key.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PublicKeyResponse:
    """Store the caller's public key, replacing any previously uploaded key.

    Replacing (rather than rejecting a second upload) intentionally supports
    a user re-registering a new device's identity key; safety-number/key-
    transparency verification of such changes is explicit future work
    (see the README threat model), not silently prevented here.
    """

    # Overwrite any previous key; the private half never left the client that generated it.
    current_user.public_key = payload.public_key
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return PublicKeyResponse(username=current_user.username, public_key=current_user.public_key)


# Look up any account's public key so a peer can derive shared session keys.
@router.get("/{username}", response_model=PublicKeyResponse)
async def get_public_key(
    # Identify which account's public key to return.
    username: str,
    # Require authentication; the spec calls this "not secret data" but still gates
    # it behind login so the endpoint cannot be used for unauthenticated account enumeration.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PublicKeyResponse:
    """Return the target account's public key, or 404 if none has been uploaded yet."""

    # current_user is unused beyond proving authentication; keep the parameter for the dependency.
    del current_user
    user = await db.scalar(select(User).where(User.username == username))
    if user is None or user.public_key is None:
        # Collapse "no such user" and "user has no key yet" into one response: neither
        # case lets a caller start an E2EE conversation, and the distinction is not useful.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="no public key available for this username"
        )
    return PublicKeyResponse(username=user.username, public_key=user.public_key)
