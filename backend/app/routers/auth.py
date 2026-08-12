"""Expose the rate-limited registration endpoint backed by Argon2id hashing."""

# Import Annotated to describe dependency-injected parameter metadata.
from typing import Annotated

# Import FastAPI's routing, dependency, HTTP-error, and request primitives.
from fastapi import APIRouter, Depends, HTTPException, Request, status

# Import SQLAlchemy's async session type and a scoped existence query helper.
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model this endpoint creates rows in.
from app.models.user import User

# Import the validated request and response shapes for this endpoint.
from app.schemas.auth import RegisterRequest, RegisterResponse

# Import the Argon2id hashing function; never hash with anything else here.
from app.security.passwords import hash_password

# Import the shared limiter that enforces the spec's rate-limiting requirement.
from app.security.rate_limit import limiter

# Group every authentication endpoint under one versionable prefix and tag.
router = APIRouter(prefix="/auth", tags=["authentication"])

# Bound registration attempts per client address; brute-force account creation
# is exactly the abuse pattern the spec's rate-limiting requirement targets.
_REGISTER_RATE_LIMIT = "5/minute"


# Register a new account with an Argon2id password hash.
@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit(_REGISTER_RATE_LIMIT)
async def register(
    # slowapi's decorator inspects this parameter to read the client address.
    request: Request,
    # Accept and validate the registration payload from the request body.
    payload: RegisterRequest,
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Create one account row from a validated registration payload.

    The public key column stays null here; Slice 3's POST /keys/me endpoint
    populates it after the client generates its X25519 keypair locally.
    """

    # Check for an existing username before attempting an insert.
    existing_username = await db.scalar(select(User.id).where(User.username == payload.username))
    # Reject a duplicate username with a specific, non-secret message.
    if existing_username is not None:
        # Return 409 Conflict rather than a generic 400 or a 500 from the database.
        raise HTTPException(status.HTTP_409_CONFLICT, detail="username is already registered")

    # Check for an existing email before attempting an insert.
    existing_email = await db.scalar(select(User.id).where(User.email == payload.email))
    # Reject a duplicate email with a specific, non-secret message.
    if existing_email is not None:
        # Return 409 Conflict rather than a generic 400 or a 500 from the database.
        raise HTTPException(status.HTTP_409_CONFLICT, detail="email is already registered")

    # Hash the plaintext password with Argon2id; the plaintext is discarded after this call.
    password_hash = hash_password(payload.password)
    # Build the new account row with no public key yet and no plaintext anywhere.
    user = User(username=payload.username, email=payload.email, password_hash=password_hash)
    # Stage the new row for insertion.
    db.add(user)
    # Persist the row, surfacing a 409 if a concurrent request won a unique-constraint race.
    try:
        # Flush and commit the transaction now so `user.id`/`created_at` are populated.
        await db.commit()
    except IntegrityError as exc:
        # Roll back the failed transaction before reporting the conflict.
        await db.rollback()
        # Treat any persistence failure here as a race on the checked-for uniqueness.
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="username or email is already registered"
        ) from exc
    # Refresh server-generated fields (id, created_at) before returning the row.
    await db.refresh(user)
    # FastAPI serializes this ORM instance through RegisterResponse's from_attributes config.
    return user
