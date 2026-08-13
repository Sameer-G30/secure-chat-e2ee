"""Define the refresh-token row that makes rotation-on-use enforceable and auditable.

Per Part B of the build plan, this table only ever stores a hash of the
issued JWT, plus `created_at`/`revoked_at`/a unique `token_hash`, never the
raw token itself and never any symmetric or private key material.
"""

# Import datetime for typed timestamp annotations.
from datetime import datetime

# Import uuid for typed primary/foreign-key annotations and default generation.
from uuid import UUID, uuid4

# Import server-side default helpers, the UUID column type, and foreign-key support.
from sqlalchemy import DateTime, ForeignKey, Index, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


# Map the refresh_tokens table exactly as scoped by Part B of the build plan.
class RefreshToken(Base):
    """Represent one issued refresh token's rotation/revocation state.

    Only a SHA-256 hash of the token is stored (see app.security.tokens);
    the raw JWT the client holds is never persisted anywhere on the server.
    """

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "refresh_tokens"

    # Identify each issued token with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Identify which account this refresh token authenticates future rotations for.
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # Store only the SHA-256 hash of the issued JWT, never the raw token.
    token_hash: Mapped[str] = mapped_column(unique=True, index=True, nullable=False)
    # Record issuance time so token age is auditable independent of the JWT's own iat claim.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Record the token's absolute expiry, mirrored from its JWT "exp" claim.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Record when rotation, logout, or reuse-detection revoked this token; null while active.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Speed up "find this user's active tokens" queries used by reuse-detection revocation.
    __table_args__ = (Index("ix_refresh_tokens_user_id", "user_id"),)
