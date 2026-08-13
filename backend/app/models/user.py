"""Define the account row the server may hold: identity, hash, and public key."""

# Import datetime for typed created_at annotations.
from datetime import datetime

# Import uuid for typed primary-key annotations and default generation.
from uuid import UUID, uuid4

# Import server-side default helpers and the portable UUID column type.
from sqlalchemy import DateTime, Uuid, func

# Import typed declarative mapping helpers introduced in SQLAlchemy 2.0.
from sqlalchemy.orm import Mapped, mapped_column

# Import the shared declarative base every table inherits.
from app.db import Base


# Map the users table exactly as scoped for this slice.
class User(Base):
    """Represent one registered account row.

    Only Argon2id password hashes and a public X25519 key ever live here.
    The server never stores a private key or plaintext password.
    """

    # Name the table explicitly rather than relying on class-name inference.
    __tablename__ = "users"

    # Identify each account with a non-guessable random UUID.
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    # Store the unique, case-sensitive handle used for login and key lookup.
    username: Mapped[str] = mapped_column(unique=True, index=True, nullable=False)
    # Store the unique account email used for registration and recovery.
    email: Mapped[str] = mapped_column(unique=True, index=True, nullable=False)
    # Store only the Argon2id hash output; the plaintext password never persists.
    password_hash: Mapped[str] = mapped_column(nullable=False)
    # Store the base64 X25519 public key uploaded by the client.
    #
    # Still nullable in Slice 3, by design rather than oversight: POST
    # /keys/me now exists and requires a bearer access token, but
    # POST /auth/register intentionally returns no tokens (its response
    # contract is exercised by Slice 2's tests and stays account-metadata
    # only). Key upload therefore happens on the client's first
    # authenticated session (immediately after the first successful
    # POST /auth/login), not at the moment the row is inserted. Adding a
    # database NOT NULL constraint would force one of two worse designs:
    # bundling key upload into registration itself (coupling account
    # creation to a client-side crypto step that can fail independently),
    # or having the client invent a placeholder key. This documents the
    # transitional rule instead: a freshly registered account is not yet
    # usable for E2EE messaging until its first login completes key
    # upload, and Slice 4's conversation/message endpoints must check
    # `public_key is not None` (both parties) before allowing a
    # conversation to start.
    public_key: Mapped[str | None] = mapped_column(nullable=True, default=None)
    # Record account creation time using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
