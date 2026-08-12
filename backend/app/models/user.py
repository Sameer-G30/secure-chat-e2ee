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
    # Nullable in this slice: the spec's POST /keys/me upload endpoint that
    # populates this column ships in Slice 3. Making the column NOT NULL now
    # would force registration to invent a placeholder key, which is worse
    # than an honest nullable column plus a Slice-3 migration that tightens
    # the constraint once every account is required to have uploaded a key.
    public_key: Mapped[str | None] = mapped_column(nullable=True, default=None)
    # Record account creation time using the database server's clock.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
