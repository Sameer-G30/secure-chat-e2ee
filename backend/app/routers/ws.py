"""Authenticate a WebSocket and relay ciphertext envelopes without decrypting them."""

# Import Annotated and UUID for dependency metadata and conversation path parameters.
from typing import Annotated
from uuid import UUID

# Import FastAPI's WebSocket primitives and HTTPException (reused for WS close reasons).
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status

# Import Pydantic's validation error type so malformed frames become protocol errors.
from pydantic import ValidationError

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the validated inbound envelope schema.
from app.schemas.messages import RelayEnvelopeIn

# Import the shared access-token verifier used by HTTP Bearer auth.
from app.security.dependencies import get_user_from_access_token

# Import persist/fan-out helpers; none of these decrypt or handle private keys.
from app.services.conversations import other_user_id
from app.services.relay import (
    EnvelopeRejected,
    authorize_relay_connection,
    connection_hub,
    relay_envelope,
)

# Group the ciphertext relay under a dedicated WebSocket prefix.
router = APIRouter(prefix="/ws", tags=["websocket"])

# Close the socket with a 4401 when the query-string access token is missing or invalid.
_WS_CLOSE_UNAUTHORIZED = 4401
# Close the socket with a 4403 when the caller is not a member / keys are missing.
_WS_CLOSE_FORBIDDEN = 4403


# Authenticate, persist, and fan out ciphertext for one 1:1 conversation.
@router.websocket("/conversations/{conversation_id}")
async def conversation_relay(
    # Accept the WebSocket upgrade from the authenticated browser tab.
    websocket: WebSocket,
    # Identify which conversation this socket is allowed to send/receive for.
    conversation_id: UUID,
    # Inject a connection-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Relay `{ciphertext, nonce, key_epoch}` between two members.

    The server never decrypts. sender_id is taken from the access token, and
    every persist is scoped by this conversation_id. Optional routing fields
    on the frame are compared against that context and rejected on mismatch.
    """

    # Browsers cannot set Authorization on WebSocket; the access token travels as a query param.
    access_token = websocket.query_params.get("access_token")
    if not access_token:
        # Refuse the upgrade rather than accepting an anonymous ciphertext socket.
        await websocket.close(code=_WS_CLOSE_UNAUTHORIZED)
        return

    try:
        # Reuse the HTTP access-token verifier so WS auth cannot drift from REST auth.
        user = await get_user_from_access_token(access_token, db)
        # Confirm membership and that both parties still have public keys.
        conversation = await authorize_relay_connection(db, conversation_id, user)
    except HTTPException as exc:
        # Map REST-style 401/403/404 onto WebSocket close codes; never include ciphertext.
        close_code = (
            _WS_CLOSE_UNAUTHORIZED
            if exc.status_code == status.HTTP_401_UNAUTHORIZED
            else _WS_CLOSE_FORBIDDEN
        )
        await websocket.close(code=close_code)
        return
    except EnvelopeRejected:
        # Fail closed when the peer row is missing; do not accept the socket.
        await websocket.close(code=_WS_CLOSE_FORBIDDEN)
        return

    # Only accept after authentication and membership checks succeed.
    await websocket.accept()
    # Track this socket so a later persist can fan the envelope to the peer.
    connection_hub.join(conversation.id, user.id, websocket)
    # Presence is metadata the server can already see; tell this tab if the peer is online.
    peer_id = other_user_id(conversation, user.id)
    await websocket.send_json(
        {
            "type": "presence",
            "user_id": str(peer_id),
            "online": connection_hub.is_connected(conversation.id, peer_id),
        }
    )
    # Tell the already-connected peer that this user is now online.
    await connection_hub.broadcast(
        conversation.id,
        {"type": "presence", "user_id": str(user.id), "online": True},
        exclude_user_id=user.id,
    )

    try:
        while True:
            # Wait for the next JSON frame from this member.
            raw_frame = await websocket.receive_json()
            # Typing frames are metadata only: never persist, never include draft text.
            if isinstance(raw_frame, dict) and raw_frame.get("type") == "typing":
                is_typing = bool(raw_frame.get("is_typing"))
                await connection_hub.broadcast(
                    conversation.id,
                    {
                        "type": "typing",
                        "user_id": str(user.id),
                        "is_typing": is_typing,
                    },
                    exclude_user_id=user.id,
                )
                continue
            try:
                # Validate ciphertext/nonce/epoch shape without attempting decryption.
                envelope = RelayEnvelopeIn.model_validate(raw_frame)
                # Persist BYTEA columns, maybe bump current_epoch, build the outbound frame.
                outbound, rotated_epoch = await relay_envelope(
                    db, conversation=conversation, sender=user, envelope=envelope
                )
            except (ValidationError, EnvelopeRejected, ValueError) as exc:
                # Tell the sender the frame was rejected; never echo ciphertext or plaintext.
                detail = (
                    "invalid envelope"
                    if isinstance(exc, ValidationError)
                    else str(exc) or "invalid envelope"
                )
                await websocket.send_json({"type": "error", "detail": detail})
                continue

            # Fan the stored envelope to the peer only; the sender already has plaintext.
            await connection_hub.broadcast(
                conversation.id,
                outbound.model_dump(mode="json"),
                exclude_user_id=user.id,
            )
            # Acknowledge persistence so the sender can attach the server-assigned id.
            await websocket.send_json({"type": "accepted", "id": str(outbound.id)})
            # Broadcast the new counter to every member, including the sender.
            if rotated_epoch is not None:
                # Metadata only: the integer clients use as the next KDF subkey id.
                await connection_hub.broadcast(
                    conversation.id,
                    {"type": "epoch", "current_epoch": rotated_epoch},
                )
    except WebSocketDisconnect:
        # Normal tab close; drop this socket from the room.
        connection_hub.leave(conversation.id, user.id, websocket)
        # Broadcast offline only when this user has no remaining sockets in the room.
        if not connection_hub.is_connected(conversation.id, user.id):
            await connection_hub.broadcast(
                conversation.id,
                {"type": "presence", "user_id": str(user.id), "online": False},
            )
    except Exception:
        # Any unexpected failure must still forget the socket; do not log envelope bytes.
        connection_hub.leave(conversation.id, user.id, websocket)
        if not connection_hub.is_connected(conversation.id, user.id):
            await connection_hub.broadcast(
                conversation.id,
                {"type": "presence", "user_id": str(user.id), "online": False},
            )
        raise
    else:
        # Leave the room if the loop ends without a disconnect exception.
        connection_hub.leave(conversation.id, user.id, websocket)
        if not connection_hub.is_connected(conversation.id, user.id):
            await connection_hub.broadcast(
                conversation.id,
                {"type": "presence", "user_id": str(user.id), "online": False},
            )
