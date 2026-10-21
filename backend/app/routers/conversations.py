"""Expose authenticated REST to start or fetch a 1:1 conversation and read its epoch."""

# Import Annotated and UUID for dependency metadata and conversation path parameters.
from typing import Annotated
from uuid import UUID

# Import FastAPI's routing, dependency, HTTP-error, and status primitives.
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the Message model for the hide endpoint's direct lookup-by-id.
from app.models.message import Message

# Import the ORM model the auth dependency returns.
from app.models.user import User

# Import the validated request and response shapes for this router's endpoints.
from app.schemas.blobs import EncryptedBlobIn, EncryptedBlobOut
from app.schemas.conversations import (
    ConversationListResponse,
    ConversationResponse,
    CreateConversationRequest,
    EpochResponse,
)
from app.schemas.messages import MessageDeletedOut, MessageHistoryResponse
from app.schemas.receipts import ConversationReadResponse

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user
from app.security.rate_limit import limiter

# Import encrypted-blob store/fetch helpers (ciphertext only; no decrypt).
from app.services.blobs import get_encrypted_blob, store_encrypted_blob
from app.services.conversations import (
    get_conversation_for_participant,
    get_conversation_response,
    get_epoch_for_participant,
    get_or_create_conversation,
    list_conversations_for_user,
    other_user_id,
)
from app.services.receipts import mark_conversation_read, peer_receipt_flags
from app.services.relay import (
    connection_hub,
    delete_message_for_everyone,
    get_message_for_sender,
    hide_all_messages_for_owner,
    hide_message_for_owner,
    list_envelopes_page,
    list_newest_envelopes_page,
    serialize_envelope,
)

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


# List 1:1 conversations the caller belongs to (sidebar), without a global message scan.
@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationListResponse:
    """Return membership-gated conversations, newest first."""

    return await list_conversations_for_user(db, current_user)


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
    # Optional page size. With `before` (or alone), this selects a newest-page window.
    # Omitted with neither cursor still returns the full conversation, oldest first.
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
    # Exclusive forward cursor: rows newer than this id, oldest-first (existing clients).
    after: Annotated[UUID | None, Query()] = None,
    # Exclusive backward cursor. Omitted with `limit` means the newest page.
    before: Annotated[UUID | None, Query()] = None,
) -> MessageHistoryResponse:
    """Return envelopes scoped by conversation_id, oldest first inside the page.

    Membership is required (non-members 404). The payload is ciphertext,
    nonce, key_epoch, sender_id, created_at, and id only — never a body
    and never a classification score. Spec §11 forbids a flat all-messages
    query with a client-side filter.

    `after` still walks forward from an older id. `before` (or `limit` with
    no cursor) returns a newest-page window. `next_cursor` on that window is
    the oldest id when older rows remain.
    """

    # The two cursors page in opposite directions; one request cannot use both.
    if after is not None and before is not None:
        # Tell the client to pick a direction instead of guessing.
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="pass only one of after or before",
        )
    # 404 if the conversation is missing or the caller is not a member.
    conversation = await get_conversation_for_participant(db, conversation_id, current_user)
    # Forward pages keep the original oldest-first `after` walk.
    if after is not None:
        # Existing callers pass `after` to continue toward newer envelopes.
        stored, next_cursor = await list_envelopes_page(
            db,
            conversation_id,
            newest_first=False,
            exclude_hidden_for=current_user.id,
            limit=limit,
            after_id=after,
        )
    # Everything else is a bounded newest-page window; unpaged reads default to 200 rows
    # so one request can never stream an unbounded conversation.
    else:
        page_limit = limit if limit is not None else 200
        # Newest page when `before` is omitted; the next older page when it is set.
        stored, next_cursor = await list_newest_envelopes_page(
            db,
            conversation_id,
            exclude_hidden_for=current_user.id,
            limit=page_limit,
            before_id=before,
        )
    # Attach the peer's delivered/read ticks onto the viewer's own sent envelopes only.
    peer_id = other_user_id(conversation, current_user.id)
    sent_ids = [row.id for row in stored if row.sender_id == current_user.id]
    flags = await peer_receipt_flags(db, peer_id=peer_id, message_ids=sent_ids)
    envelopes = []
    for row in stored:
        delivered, read = (
            flags.get(row.id, (False, False))
            if row.sender_id == current_user.id
            else (False, False)
        )
        envelopes.append(serialize_envelope(row, peer_delivered=delivered, peer_read=read))
    return MessageHistoryResponse(
        messages=envelopes,
        next_cursor=next_cursor,
    )


# Hard-delete one message for every participant ("delete for everyone").
@router.delete(
    "/conversations/{conversation_id}/messages/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation_message(
    # Identify which conversation the message belongs to (routing/scoping only).
    conversation_id: UUID,
    # Identify which stored envelope to remove.
    message_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Remove a message the caller sent, and tell the peer it is gone.

    Sender-only: a 403 (not a 404) tells a member of the conversation who is
    not the sender that the row exists but they cannot delete it, matching
    how the rest of this API reports authorization failures. A hard delete,
    unlike the legacy app's buggy soft-delete flag the UI never actually read.
    """

    # 404 if the conversation is missing or the caller is not a member.
    await get_conversation_for_participant(db, conversation_id, current_user)
    # 404 if the message is not in this conversation; 403 if the caller did not send it.
    message = await get_message_for_sender(db, conversation_id, message_id, current_user.id)
    await delete_message_for_everyone(db, message)
    # Tell the peer (never the sender, who already knows) that this row is gone.
    await connection_hub.broadcast(
        conversation_id,
        MessageDeletedOut(conversation_id=conversation_id, id=message_id).model_dump(mode="json"),
        exclude_user_id=current_user.id,
    )


# Hide one message from the caller's own history only ("delete for me").
@router.post(
    "/conversations/{conversation_id}/messages/{message_id}/hide",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def hide_conversation_message(
    # Identify which conversation the message belongs to (routing/scoping only).
    conversation_id: UUID,
    # Identify which stored envelope to hide from the caller's own view.
    message_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Hide a message from the caller's own future `GET .../messages` calls only.

    Either participant may hide either party's message from their own view —
    unlike delete-for-everyone, this needs no sender check. The peer's copy
    of the same envelope, and the peer's own history, are never affected.
    """

    # 404 if the conversation is missing or the caller is not a member.
    await get_conversation_for_participant(db, conversation_id, current_user)
    # 404 (not 403) if the message is missing or outside this conversation; either
    # member may hide either party's message from their own view.
    message_row = await db.get(Message, message_id)
    if message_row is None or message_row.conversation_id != conversation_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="message not found")
    await hide_message_for_owner(db, message_row, current_user.id)


# Hide every envelope in this conversation from the caller's own history only.
@router.post(
    "/conversations/{conversation_id}/hide-all",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def hide_all_conversation_messages(
    # Identify which conversation to clear from the caller's own view.
    conversation_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Hide every message in this conversation for the caller only.

    Does not delete message rows, blobs, or the conversation. Does not
    broadcast message_deleted. The response body is empty (no ciphertext).
    """

    # 404 if the conversation is missing or the caller is not a member.
    await get_conversation_for_participant(db, conversation_id, current_user)
    # Insert message_hides for this owner. The peer's GET is unchanged.
    await hide_all_messages_for_owner(db, conversation_id, current_user.id)


# Mark this conversation read for the caller and ack inbound envelopes as read.
@router.post("/conversations/{conversation_id}/read", response_model=ConversationReadResponse)
async def mark_conversation_as_read(
    # Identify which conversation to mark read.
    conversation_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationReadResponse:
    """Upsert the last-read cursor and fan delivered/read ticks to the peer.

    This is metadata the server can already infer from WebSocket activity.
    It never stores a preview string or a message body.
    """

    conversation = await get_conversation_for_participant(db, conversation_id, current_user)
    last_read_at, frames = await mark_conversation_read(
        db, conversation=conversation, viewer=current_user
    )
    for frame in frames:
        await connection_hub.broadcast(
            conversation.id,
            frame,
            exclude_user_id=current_user.id,
        )
    return ConversationReadResponse(conversation_id=conversation.id, last_read_at=last_read_at)


# Store one client-encrypted image blob for a conversation the caller belongs to.
@router.post(
    "/conversations/{conversation_id}/blobs",
    response_model=EncryptedBlobOut,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("30/minute")
async def upload_conversation_blob(
    # slowapi reads the client address from this parameter.
    request: Request,
    # Identify which conversation this sealed file belongs to.
    conversation_id: UUID,
    # Accept opaque ciphertext, nonce, and a client-chosen blob id.
    payload: EncryptedBlobIn,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EncryptedBlobOut:
    """Persist sealed file bytes. The server never opens the ciphertext."""

    conversation = await get_conversation_for_participant(db, conversation_id, current_user)
    return await store_encrypted_blob(
        db, conversation=conversation, uploader=current_user, payload=payload
    )


# Return one sealed blob that belongs to a conversation the caller is in.
@router.get(
    "/conversations/{conversation_id}/blobs/{blob_id}",
    response_model=EncryptedBlobOut,
)
async def download_conversation_blob(
    # Identify which conversation this sealed file belongs to.
    conversation_id: UUID,
    # Identify the client-chosen blob id.
    blob_id: UUID,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EncryptedBlobOut:
    """Return opaque ciphertext+nonce. Non-members receive 404."""

    conversation = await get_conversation_for_participant(db, conversation_id, current_user)
    return await get_encrypted_blob(db, conversation=conversation, blob_id=blob_id)
