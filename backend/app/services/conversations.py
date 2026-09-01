"""Load, create, and authorize 1:1 conversations without touching message plaintext."""

# Import UUID for typed identity comparisons and path parameters.
from uuid import UUID

# Import FastAPI's HTTP-error primitives so callers can return stable status codes.
from fastapi import HTTPException, status

# Import SQLAlchemy query helpers and the uniqueness-conflict error.
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM models this service reads and writes.
from app.models.conversation import Conversation
from app.models.user import User

# Import the response shapes so routers do not rebuild participant payloads.
from app.schemas.conversations import (
    ConversationListResponse,
    ConversationParticipant,
    ConversationResponse,
    EpochResponse,
)

# Shared detail when a caller is not a member or the conversation does not exist.
_CONVERSATION_NOT_FOUND_DETAIL = "conversation not found"
# Shared detail when the peer cannot start E2EE (unknown account or no public key yet).
_PEER_UNAVAILABLE_DETAIL = "no public key available for this username"
# Shared detail when the authenticated caller has not uploaded a public key yet.
_SELF_MISSING_KEY_DETAIL = "upload a public key before starting a conversation"
# Shared detail when both parties are the same account.
_SELF_CHAT_DETAIL = "cannot start a conversation with yourself"


# Order two account UUIDs so user_a_id is always strictly less than user_b_id.
def ordered_user_ids(left: UUID, right: UUID) -> tuple[UUID, UUID]:
    """Return (smaller, larger) so the CHECK (user_a_id < user_b_id) constraint holds."""

    # Compare UUID integers; this matches PostgreSQL's UUID ordering for the CHECK.
    return (left, right) if left < right else (right, left)


# Return the other participant's id given one member of a 1:1 conversation.
def other_user_id(conversation: Conversation, user_id: UUID) -> UUID:
    """Return the peer UUID for a caller who is already a known member."""

    # The caller is whichever side of the canonical pair matches user_id.
    if conversation.user_a_id == user_id:
        return conversation.user_b_id
    return conversation.user_a_id


# Refuse to proceed unless both accounts have uploaded an X25519 public key.
def require_both_public_keys(self_user: User, peer_user: User) -> None:
    """Raise if either participant still has a null public_key.

    users.public_key stays nullable by design (Slice 3 transitional rule).
    Conversation and message endpoints are the enforcement point: E2EE
    cannot start until both parties have published a public key.
    """

    # The caller can fix a missing own key; tell them that directly.
    if self_user.public_key is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail=_SELF_MISSING_KEY_DETAIL)
    # Collapse "no such peer key" with lookup-style 404 so this is not a user oracle.
    if peer_user.public_key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_PEER_UNAVAILABLE_DETAIL)


# Load a conversation only when the caller is one of its two members.
async def get_conversation_for_participant(
    db: AsyncSession,
    conversation_id: UUID,
    user: User,
) -> Conversation:
    """Return the conversation row, or 404 if it is missing or the caller is not a member."""

    # Scope the lookup by conversation_id *and* membership so a guessed UUID leaks nothing.
    conversation = await db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            or_(Conversation.user_a_id == user.id, Conversation.user_b_id == user.id),
        )
    )
    if conversation is None:
        # Do not distinguish "does not exist" from "exists but you are not in it".
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_CONVERSATION_NOT_FOUND_DETAIL)
    return conversation


# Build the client-facing conversation payload from ORM rows.
def serialize_conversation(
    conversation: Conversation,
    self_user: User,
    peer_user: User,
) -> ConversationResponse:
    """Return ids, usernames, public keys, and the non-secret epoch counter."""

    return ConversationResponse(
        id=conversation.id,
        current_epoch=conversation.current_epoch,
        created_at=conversation.created_at,
        self=ConversationParticipant(
            id=self_user.id,
            username=self_user.username,
            public_key=self_user.public_key,
        ),
        peer=ConversationParticipant(
            id=peer_user.id,
            username=peer_user.username,
            public_key=peer_user.public_key,
        ),
    )


# Start or fetch the unique 1:1 conversation between the caller and a named peer.
async def get_or_create_conversation(
    db: AsyncSession,
    self_user: User,
    peer_username: str,
) -> ConversationResponse:
    """Return the canonical conversation for this pair, creating it when needed.

    Both parties must already have a non-null public_key. The pair is stored
    with user_a_id < user_b_id so initiator identity cannot duplicate the row.
    """

    # A conversation with oneself has no well-defined crypto_kx pairing.
    if peer_username == self_user.username:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=_SELF_CHAT_DETAIL)

    # Look up the peer by the same unique handle GET /keys/{username} uses.
    peer_user = await db.scalar(select(User).where(User.username == peer_username))
    if peer_user is None:
        # Collapse unknown usernames into the same 404 used when a key is missing.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_PEER_UNAVAILABLE_DETAIL)

    # Enforce the Slice 3 transitional rule: nullable public_key is not yet usable for E2EE.
    require_both_public_keys(self_user, peer_user)

    # Canonicalize pair order before any insert so the CHECK constraint is satisfied.
    user_a_id, user_b_id = ordered_user_ids(self_user.id, peer_user.id)

    # Reuse an existing row when this pair has already talked.
    existing = await db.scalar(
        select(Conversation).where(
            Conversation.user_a_id == user_a_id,
            Conversation.user_b_id == user_b_id,
        )
    )
    if existing is not None:
        return serialize_conversation(existing, self_user, peer_user)

    # Insert the canonical pair; a concurrent request may win the unique constraint.
    conversation = Conversation(user_a_id=user_a_id, user_b_id=user_b_id, current_epoch=0)
    db.add(conversation)
    try:
        await db.commit()
    except IntegrityError:
        # Another request created the same pair first; load that winner instead of 500ing.
        await db.rollback()
        existing = await db.scalar(
            select(Conversation).where(
                Conversation.user_a_id == user_a_id,
                Conversation.user_b_id == user_b_id,
            )
        )
        if existing is None:
            # The unique-constraint race should always leave a row; fail closed if not.
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="could not create conversation",
            ) from None
        return serialize_conversation(existing, self_user, peer_user)

    await db.refresh(conversation)
    return serialize_conversation(conversation, self_user, peer_user)


# Load a conversation the caller belongs to and return its serialized form.
async def get_conversation_response(
    db: AsyncSession,
    conversation_id: UUID,
    self_user: User,
) -> ConversationResponse:
    """Return a membership-gated conversation payload, enforcing both-keys on read too."""

    conversation = await get_conversation_for_participant(db, conversation_id, self_user)
    peer_user = await db.get(User, other_user_id(conversation, self_user.id))
    if peer_user is None:
        # The foreign key should make this impossible; fail closed rather than serializing holes.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_CONVERSATION_NOT_FOUND_DETAIL)
    # Message/conversation endpoints must still reject a party whose public_key is None.
    require_both_public_keys(self_user, peer_user)
    return serialize_conversation(conversation, self_user, peer_user)


# Return the non-secret epoch counter for a conversation the caller belongs to.
async def get_epoch_for_participant(
    db: AsyncSession,
    conversation_id: UUID,
    self_user: User,
) -> EpochResponse:
    """Return current_epoch after confirming membership and both public keys."""

    conversation = await get_conversation_for_participant(db, conversation_id, self_user)
    peer_user = await db.get(User, other_user_id(conversation, self_user.id))
    if peer_user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_CONVERSATION_NOT_FOUND_DETAIL)
    require_both_public_keys(self_user, peer_user)
    return EpochResponse(conversation_id=conversation.id, current_epoch=conversation.current_epoch)


# List every 1:1 conversation the caller is a member of, newest conversation first.
async def list_conversations_for_user(
    db: AsyncSession, self_user: User
) -> ConversationListResponse:
    """Return membership-gated conversation payloads without scanning all messages."""

    rows = await db.scalars(
        select(Conversation)
        .where(or_(Conversation.user_a_id == self_user.id, Conversation.user_b_id == self_user.id))
        .order_by(Conversation.created_at.desc())
    )
    conversations: list[ConversationResponse] = []
    for conversation in rows:
        peer_user = await db.get(User, other_user_id(conversation, self_user.id))
        if peer_user is None:
            continue
        conversations.append(serialize_conversation(conversation, self_user, peer_user))
    return ConversationListResponse(conversations=conversations)
