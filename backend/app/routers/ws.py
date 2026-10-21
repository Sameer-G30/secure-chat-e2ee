"""Authenticate a WebSocket and relay ciphertext envelopes without decrypting them."""

# Import json for explicit, size-checked frame parsing.
import json

# Import Annotated and UUID for dependency metadata and conversation path parameters.
from typing import Annotated
from uuid import UUID

# Import FastAPI's WebSocket primitives and HTTPException (reused for WS close reasons).
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)

# Import Pydantic's validation error type so malformed frames become protocol errors.
from pydantic import ValidationError

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the shared access-token verifier used by HTTP Bearer auth.
from app.config import get_settings

# Import the request-scoped database session dependency.
from app.db import get_db
from app.models.user import User

# Import the validated inbound envelope schema.
from app.schemas.messages import RelayEnvelopeIn

# Import the validated inbound receipt schema (metadata only, never a message body).
from app.schemas.receipts import ReceiptFrameIn
from app.security.dependencies import get_current_user, get_user_from_access_token
from app.security.rate_limit import limiter

# Import persist/fan-out helpers; none of these decrypt or handle private keys.
from app.services.conversations import other_user_id
from app.services.receipts import apply_receipt_batch
from app.services.relay import (
    EnvelopeRejected,
    EnvelopeSilentlyDropped,
    authorize_relay_connection,
    connection_hub,
    relay_envelope,
)
from app.services.ws_tickets import issue_ticket, redeem_ticket

# Group the ciphertext relay under a dedicated WebSocket prefix.
router = APIRouter(prefix="/ws", tags=["websocket"])

# Close the socket with a 4401 when the query-string access token is missing or invalid.
_WS_CLOSE_UNAUTHORIZED = 4401
# Close the socket with a 4403 when the caller is not a member / keys are missing.
_WS_CLOSE_FORBIDDEN = 4403
# Largest text frame accepted; ciphertext is capped at 32 KiB (base64 ~43 KiB) plus JSON overhead.
_MAX_FRAME_CHARS = 131072


# Exchange a valid bearer token for a 30-second single-use WebSocket ticket.
@router.post("/conversations/{conversation_id}/ticket")
@limiter.limit("60/minute")
async def create_ws_ticket(
    # Required by slowapi to derive the client key.
    request: Request,
    conversation_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    """Return a ticket only to members of the conversation."""

    # Membership, key, and block checks match the WebSocket handshake exactly.
    try:
        await authorize_relay_connection(db, conversation_id, user)
    except EnvelopeRejected as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="conversation unavailable") from exc
    return {"ticket": issue_ticket(user.id, conversation_id), "expires_in": 30}


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

    settings = get_settings()
    # Reject cross-site pages: browsers always send Origin on WebSocket upgrades.
    origin = websocket.headers.get("origin")
    if origin is not None and origin != settings.frontend_origin:
        await websocket.close(code=_WS_CLOSE_FORBIDDEN)
        return

    # Preferred auth: single-use ticket from POST .../ticket (keeps JWTs out of URLs and logs).
    ticket = websocket.query_params.get("ticket")
    access_token = websocket.query_params.get("access_token")
    # The legacy query-string token is only honoured outside production.
    if access_token and not (settings.app_env == "development" or settings.ws_allow_query_token):
        access_token = None
    if not ticket and not access_token:
        # Refuse the upgrade rather than accepting an anonymous ciphertext socket.
        await websocket.close(code=_WS_CLOSE_UNAUTHORIZED)
        return

    try:
        if ticket:
            ticket_user_id = redeem_ticket(ticket, conversation_id)
            if ticket_user_id is None:
                await websocket.close(code=_WS_CLOSE_UNAUTHORIZED)
                return
            loaded = await db.get(User, ticket_user_id)
            if loaded is None:
                await websocket.close(code=_WS_CLOSE_UNAUTHORIZED)
                return
            user = loaded
        else:
            # Reuse the HTTP access-token verifier so WS auth cannot drift from REST auth.
            assert access_token is not None
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
            raw_text = await websocket.receive_text()
            # Oversized frames are a DoS vector; close before parsing.
            if len(raw_text) > _MAX_FRAME_CHARS:
                await websocket.close(code=1009)
                break
            try:
                raw_frame = json.loads(raw_text)
            except ValueError:
                await websocket.send_json({"type": "error", "detail": "invalid frame"})
                continue
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
            # Receipt frames are metadata only: delivered/read ticks, never a message body.
            if isinstance(raw_frame, dict) and raw_frame.get("type") == "receipt":
                try:
                    receipt = ReceiptFrameIn.model_validate(raw_frame)
                    frames = await apply_receipt_batch(
                        db,
                        conversation=conversation,
                        recipient=user,
                        kind=receipt.kind,
                        message_ids=receipt.message_ids,
                    )
                except (ValidationError, HTTPException, ValueError):
                    await websocket.send_json({"type": "error", "detail": "invalid receipt"})
                    continue
                for frame in frames:
                    await connection_hub.broadcast(
                        conversation.id,
                        frame,
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
            except EnvelopeSilentlyDropped as dropped:
                # The recipient has blocked this sender. Fake-ack so the composer does
                # not show a stuck-sending state, but never persist and never broadcast
                # — the block's existence is not disclosed to either party.
                await websocket.send_json({"type": "accepted", "id": str(dropped.fake_id)})
                continue
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
            await websocket.send_json(
                {
                    "type": "accepted",
                    "id": str(outbound.id),
                    "created_at": outbound.created_at,
                }
            )
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
