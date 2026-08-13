"""Expose the rate-limited registration, login, refresh, and logout endpoints."""

# Import datetime helpers to compare token expiry against the current time.
from datetime import UTC, datetime

# Import Annotated to describe dependency-injected parameter metadata.
from typing import Annotated

# Import FastAPI's routing, dependency, HTTP-error, and request primitives.
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

# Import SQLAlchemy's async session type and scoped query/update helpers.
from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM models this router reads and writes.
from app.models.refresh_token import RefreshToken
from app.models.user import User

# Import the validated request and response shapes for this router's endpoints.
from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    RegisterResponse,
    TokenPairResponse,
)

# Import the Argon2id hashing/verification functions; never use anything weaker here.
from app.security.passwords import hash_password, verify_password

# Import the shared limiter that enforces the spec's rate-limiting requirement.
from app.security.rate_limit import limiter

# Import token issuance/verification; PyJWT per decision A4, not python-jose.
from app.security.tokens import (
    REFRESH_TOKEN_TYPE,
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_refresh_token,
)

# Group every authentication endpoint under one versionable prefix and tag.
router = APIRouter(prefix="/auth", tags=["authentication"])

# Bound registration attempts per client address; brute-force account creation
# is exactly the abuse pattern the spec's rate-limiting requirement targets.
_REGISTER_RATE_LIMIT = "5/minute"
# Bound login attempts per client address; this is the spec's other explicitly
# required rate limit and the classic target of credential-stuffing/brute force.
_LOGIN_RATE_LIMIT = "10/minute"
# Generic, non-specific message for every failed login cause (wrong username,
# wrong password, or an unusable account). Never reveal which one it was.
_INVALID_CREDENTIALS_DETAIL = "invalid username or password"


# Normalize a possibly timezone-naive datetime read back from the database.
def _as_utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime; SQLite drops tzinfo that Postgres keeps."""

    # SQLite's DateTime(timezone=True) silently stores naive values; Postgres does not.
    # Every value this project writes is already UTC, so naive values are safely UTC too.
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


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


# Authenticate an existing account and issue a fresh access/refresh token pair.
@router.post("/login", response_model=TokenPairResponse)
@limiter.limit(_LOGIN_RATE_LIMIT)
async def login(
    # slowapi's decorator inspects this parameter to read the client address.
    request: Request,
    # Accept and validate the login payload from the request body.
    payload: LoginRequest,
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenPairResponse:
    """Verify Argon2id credentials and issue a new access/refresh token pair.

    The response never distinguishes "no such username" from "wrong
    password": both cases return the same 401 with the same generic detail,
    so the endpoint cannot be used to enumerate registered usernames.
    """

    # Look up the account; a missing row is handled identically to a wrong password below.
    user = await db.scalar(select(User).where(User.username == payload.username))
    # Verify in constant time via Argon2's own comparison, never a manual string compare.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=_INVALID_CREDENTIALS_DETAIL)

    # Issue a short-lived access token bound to this account.
    access_token = create_access_token(user.id, user.username)
    # Issue a longer-lived refresh token; only its hash is persisted below.
    issued_refresh = create_refresh_token(user.id)
    # Persist the refresh token's hash, never the raw token itself.
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=issued_refresh.token_hash,
            expires_at=issued_refresh.expires_at,
        )
    )
    await db.commit()

    return TokenPairResponse(
        access_token=access_token,
        refresh_token=issued_refresh.raw_token,
        has_public_key=user.public_key is not None,
    )


# Rotate a refresh token for a new access/refresh pair, detecting reuse of a stale token.
@router.post("/refresh", response_model=TokenPairResponse)
async def refresh(
    # Accept and validate the refresh payload from the request body.
    payload: RefreshRequest,
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenPairResponse:
    """Verify, rotate, and reissue a refresh token exactly once per token.

    Rotation-on-use: presenting a refresh token always revokes it, whether
    or not the request succeeds. Presenting an *already-revoked* token is
    treated as evidence the token was stolen and replayed, so every other
    active refresh token for that account is revoked too, forcing the
    legitimate user to log in again on every device.
    """

    # Verify the JWT's own signature and expiry before touching the database at all.
    try:
        decode_token(payload.refresh_token, expected_type=REFRESH_TOKEN_TYPE)
    except TokenError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="invalid or expired refresh token"
        ) from exc

    # Look up the persisted record by the token's hash, never by its raw value.
    token_hash = hash_refresh_token(payload.refresh_token)
    stored_token = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if stored_token is None:
        # A syntactically valid JWT the server never issued (or already deleted).
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="refresh token is not recognized")

    if stored_token.revoked_at is not None:
        # This exact token was already rotated or logged out once before.
        # Treat resubmission as token theft/replay: burn every other active
        # token for this account so a stolen-and-reused token cannot be
        # used to maintain silent access alongside the legitimate user.
        await db.execute(
            sa_update(RefreshToken)
            .where(RefreshToken.user_id == stored_token.user_id)
            .where(RefreshToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(UTC))
        )
        await db.commit()
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="refresh token has already been used; all sessions were revoked",
        )

    if _as_utc(stored_token.expires_at) < datetime.now(UTC):
        # Expired tokens are inert; reject without treating this as reuse.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="refresh token has expired")

    # Load the account this token belongs to.
    user = await db.get(User, stored_token.user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="account no longer exists")

    # Rotate: this presented token becomes permanently unusable regardless of outcome.
    stored_token.revoked_at = datetime.now(UTC)
    # Issue the replacement access token.
    access_token = create_access_token(user.id, user.username)
    # Issue the replacement refresh token.
    issued_refresh = create_refresh_token(user.id)
    # Persist the new refresh token's hash, never the raw token itself.
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=issued_refresh.token_hash,
            expires_at=issued_refresh.expires_at,
        )
    )
    await db.commit()

    return TokenPairResponse(
        access_token=access_token,
        refresh_token=issued_refresh.raw_token,
        has_public_key=user.public_key is not None,
    )


# Revoke one refresh token so a logged-out session cannot later be rotated.
@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    # Accept the same refresh-token payload shape used by rotation.
    payload: RefreshRequest,
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Revoke the presented refresh token if it exists and is still active.

    Logout intentionally never errors on an unknown or already-revoked
    token: the caller's goal (this token must not work afterward) is
    already satisfied in both cases, and distinguishing them would leak
    information for no benefit.
    """

    token_hash = hash_refresh_token(payload.refresh_token)
    stored_token = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if stored_token is not None and stored_token.revoked_at is None:
        stored_token.revoked_at = datetime.now(UTC)
        await db.commit()
    # Return no body; 204 tells the client the logout request itself succeeded.
    return Response(status_code=status.HTTP_204_NO_CONTENT)
