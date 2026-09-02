"""Store and load client-encrypted file bytes without opening them."""

# Import base64 to decode inbound wire format and re-encode outbound frames.
import base64

# Import UUID for the client-chosen blob identity.
from uuid import UUID

# Import FastAPI's HTTP-error primitives so membership misses stay 404.
from fastapi import HTTPException, status

# Import SQLAlchemy query helpers used for conversation-scoped blob reads.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM models this service reads and writes.
from app.models.conversation import Conversation
from app.models.encrypted_blob import EncryptedBlob
from app.models.user import User

# Import the validated upload shape and the download shape.
from app.schemas.blobs import EncryptedBlobIn, EncryptedBlobOut


# Persist one client-chosen sealed blob for a conversation the uploader belongs to.
async def store_encrypted_blob(
    db: AsyncSession,
    *,
    conversation: Conversation,
    uploader: User,
    payload: EncryptedBlobIn,
) -> EncryptedBlobOut:
    """Insert opaque ciphertext+nonce. Duplicate ids conflict with 409."""

    # Reject a colliding client-chosen id rather than overwriting another member's blob.
    existing = await db.scalar(
        select(EncryptedBlob).where(
            EncryptedBlob.id == payload.id,
            EncryptedBlob.conversation_id == conversation.id,
        )
    )
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="blob already exists")
    # Decode wire base64 into BYTEA columns; lengths were already validated.
    ciphertext = base64.b64decode(payload.ciphertext, validate=True)
    nonce = base64.b64decode(payload.nonce, validate=True)
    stored = EncryptedBlob(
        id=payload.id,
        conversation_id=conversation.id,
        uploader_id=uploader.id,
        ciphertext=ciphertext,
        nonce=nonce,
        byte_length=len(ciphertext),
    )
    db.add(stored)
    await db.commit()
    await db.refresh(stored)
    return serialize_encrypted_blob(stored)


# Load one sealed blob that belongs to a conversation the caller is in.
async def get_encrypted_blob(
    db: AsyncSession,
    *,
    conversation: Conversation,
    blob_id: UUID,
) -> EncryptedBlobOut:
    """Return opaque ciphertext+nonce, or 404 if the blob is missing from this chat."""

    stored = await db.scalar(
        select(EncryptedBlob).where(
            EncryptedBlob.id == blob_id,
            EncryptedBlob.conversation_id == conversation.id,
        )
    )
    if stored is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="blob not found")
    return serialize_encrypted_blob(stored)


# Build the client-facing sealed-blob payload from an ORM row.
def serialize_encrypted_blob(stored: EncryptedBlob) -> EncryptedBlobOut:
    """Return id, ciphertext, nonce, and metadata; never opened file bytes."""

    return EncryptedBlobOut(
        id=stored.id,
        conversation_id=stored.conversation_id,
        uploader_id=stored.uploader_id,
        ciphertext=base64.b64encode(stored.ciphertext).decode("ascii"),
        nonce=base64.b64encode(stored.nonce).decode("ascii"),
        byte_length=stored.byte_length,
        created_at=stored.created_at,
    )
