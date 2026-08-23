"""Expose authenticated REST to start or fetch a 1:1 conversation and read its epoch."""

# Import Annotated and UUID for dependency metadata and conversation path parameters.
from typing import Annotated
from uuid import UUID

# Import FastAPI's routing, dependency, and HTTP-error primitives.
from fastapi import APIRouter, Depends

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model the auth dependency returns.
from app.models.user import User

# Import the validated request and response shapes for this router's endpoints.
from app.schemas.conversations import (
    ConversationResponse,
    CreateConversationRequest,
    EpochResponse,
)
from app.schemas.messages import MessageHistoryResponse

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user

# Import conversation load/create helpers so routers stay thin.
from app.services.conversations import (
    get_conversation_for_participant,
    get_conversation_response,
    get_epoch_for_participant,
    get_or_create_conversation,
)
from app.services.relay import list_envelopes_for_conversation, serialize_envelope

# Group conversation REST under one versionable tag; paths are absolute so
# POST /conversations is not redirected to POST /conversations/ (which would drop the body).
router = APIRouter(tags=["conversations"])


# Start or fetch the unique 1:1 conversation between the caller and a named peer.
@router.post("/conversations", response_model=ConversationResponse)
async def start_or_fetch_conversation(
    # Accept the peer's username; the pair is canonicalized server-side.
    payload: CreateConversationRequest,
    # Require a valid access token; only signed-in users may start a conversation.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationResponse:
    """Return the canonical conversation for this pair, creating it when needed.

    Both parties must have a non-null public_key. users.public_key stays
    nullable in the schema; this endpoint is the enforcement point.
    """

    return await get_or_create_conversation(db, current_user, payload.peer_username)


# Fetch an existing conversation by id, only if the caller is a member.
@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def fetch_conversation(
    # Identify which conversation to return.
    conversation_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationResponse:
    """Return conversation membership and epoch for a participating caller."""

    return await get_conversation_response(db, conversation_id, current_user)


# Return the non-secret epoch counter for a conversation the caller belongs to.
@router.get("/conversations/{conversation_id}/epoch", response_model=EpochResponse)
async def fetch_conversation_epoch(
    # Identify which conversation's counter to return.
    conversation_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EpochResponse:
    """Return current_epoch, a non-secret integer used only as a KDF subkey id.

    Equivalent to spec §6.4's GET /keys/conversations/{id}/epoch; that alias
    lives on the keys router so clients can use either path.
    """

    return await get_epoch_for_participant(db, conversation_id, current_user)


# Return ciphertext-only envelopes for one conversation the caller belongs to.
@router.get("/conversations/{conversation_id}/messages", response_model=MessageHistoryResponse)
async def fetch_conversation_messages(
    # Identify which conversation's envelopes to return.
    conversation_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageHistoryResponse:
    """Return envelopes scoped by conversation_id, oldest first.

    Membership is required (non-members 404). The payload is ciphertext,
    nonce, key_epoch, sender_id, created_at, and id only — never a body
    and never a classification score. Spec §11 forbids a flat all-messages
    query with a client-side filter.
    """

    # 404 if the conversation is missing or the caller is not a member.
    await get_conversation_for_participant(db, conversation_id, current_user)
    # Read only this conversation's rows, oldest first for the transcript.
    stored = await list_envelopes_for_conversation(db, conversation_id, newest_first=False)
    return MessageHistoryResponse(messages=[serialize_envelope(row) for row in stored])