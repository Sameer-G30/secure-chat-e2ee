"""Verify conversation create/lookup constraints, epoch reads, auth, and key gates."""

# Import base64 to build well-formed public-key payloads for test accounts.
import base64

# Import UUID to inspect canonical pair ordering on stored conversation rows.
from uuid import UUID

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import SQLAlchemy helpers to assert constraints and conversation-scoped queries.
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# Import ORM models used for direct constraint and scoping assertions.
from app.models.conversation import Conversation
from app.models.user import User

# Import the conversation-scoped envelope reader the WebSocket path also uses.
from app.services.conversations import ordered_user_ids
from app.services.relay import list_envelopes_for_conversation, persist_envelope

_ALICE = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "correct horse battery staple",
}
_BOB = {
    "username": "bob",
    "email": "bob@example.com",
    "password": "another strong passphrase",
}
_CAROL = {
    "username": "carol",
    "email": "carol@example.com",
    "password": "carol has a strong passphrase",
}


# Build a syntactically valid base64 X25519-sized (32-byte) public key for tests.
def _fake_public_key(fill_byte: int = 0x01) -> str:
    """Return base64 text decoding to exactly 32 bytes, as a real key would."""

    return base64.b64encode(bytes([fill_byte]) * 32).decode("ascii")


# Register, log in, and optionally upload a public key, returning the access token.
async def _register_login(
    client: AsyncClient,
    payload: dict[str, str],
    *,
    upload_key: bool = True,
    key_fill: int = 0x01,
) -> str:
    """Create an account, log in, and optionally complete key upload."""

    await client.post("/auth/register", json=payload)
    response = await client.post(
        "/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    access_token = str(response.json()["access_token"])
    if upload_key:
        await client.post(
            "/keys/me",
            json={"public_key": _fake_public_key(key_fill)},
            headers={"Authorization": f"Bearer {access_token}"},
        )
    return access_token


# Confirm two keyed accounts can start a conversation and get a stable id back.
async def test_start_conversation_returns_canonical_pair(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Create a conversation from Alice to Bob and require user_a_id < user_b_id."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    response = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["self"]["username"] == "alice"
    assert body["peer"]["username"] == "bob"
    assert body["peer"]["public_key"] == _fake_public_key(0x02)
    assert body["current_epoch"] == 0

    conversation = await db_session.scalar(
        select(Conversation).where(Conversation.id == UUID(body["id"]))
    )
    assert conversation is not None
    assert conversation.user_a_id < conversation.user_b_id
    alice = await db_session.scalar(select(User).where(User.username == "alice"))
    bob = await db_session.scalar(select(User).where(User.username == "bob"))
    assert alice is not None and bob is not None
    expected_a, expected_b = ordered_user_ids(alice.id, bob.id)
    assert conversation.user_a_id == expected_a
    assert conversation.user_b_id == expected_b


# Confirm initiator identity cannot duplicate the pair; the same row is returned.
async def test_start_conversation_is_idempotent_regardless_of_who_initiates(
    client: AsyncClient,
) -> None:
    """Have Alice then Bob start the same pair and require one conversation id."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    bob_token = await _register_login(client, _BOB, key_fill=0x02)
    alice_started = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    bob_started = await client.post(
        "/conversations",
        json={"peer_username": "alice"},
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert alice_started.status_code == 200
    assert bob_started.status_code == 200
    assert alice_started.json()["id"] == bob_started.json()["id"]


# Confirm the CHECK constraint rejects an unordered pair inserted behind the API.
async def test_conversation_check_constraint_rejects_unordered_ids(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Insert user_a_id > user_b_id directly and require the database to reject it."""

    await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    alice = await db_session.scalar(select(User).where(User.username == "alice"))
    bob = await db_session.scalar(select(User).where(User.username == "bob"))
    assert alice is not None and bob is not None
    larger, smaller = (alice.id, bob.id) if alice.id > bob.id else (bob.id, alice.id)
    db_session.add(Conversation(user_a_id=larger, user_b_id=smaller, current_epoch=0))
    try:
        await db_session.commit()
        raise AssertionError("expected CHECK (user_a_id < user_b_id) to reject the insert")
    except IntegrityError:
        await db_session.rollback()


# Confirm a conversation with yourself is rejected before any row is written.
async def test_cannot_start_conversation_with_self(client: AsyncClient) -> None:
    """Ask Alice to chat with Alice and require a 400."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    response = await client.post(
        "/conversations",
        json={"peer_username": "alice"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 400


# Confirm the caller must have uploaded a public key before starting a conversation.
async def test_conversation_rejects_caller_without_public_key(client: AsyncClient) -> None:
    """Leave Alice's public_key null and require POST /conversations to 403."""

    alice_token = await _register_login(client, _ALICE, upload_key=False)
    await _register_login(client, _BOB, key_fill=0x02)
    response = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 403


# Confirm the peer must also have a public key; unknown and keyless collapse to 404.
async def test_conversation_rejects_peer_without_public_key(client: AsyncClient) -> None:
    """Register Bob without a key and require Alice's create attempt to 404."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, upload_key=False)
    response = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "no public key available for this username"


# Confirm an unknown peer username uses the same 404 as a keyless peer.
async def test_conversation_rejects_unknown_peer_with_same_404(client: AsyncClient) -> None:
    """Ask to chat with a username that was never registered."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    response = await client.post(
        "/conversations",
        json={"peer_username": "nobody"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "no public key available for this username"


# Confirm conversation create/fetch requires a bearer access token.
async def test_conversation_create_requires_authentication(client: AsyncClient) -> None:
    """POST /conversations with no Authorization header and require 401 or 403."""

    response = await client.post("/conversations", json={"peer_username": "bob"})
    assert response.status_code in (401, 403)


# Confirm GET by id returns the same conversation to a member and 404s a stranger.
async def test_fetch_conversation_is_membership_gated(client: AsyncClient) -> None:
    """Let Carol guess Alice/Bob's conversation id and require a 404, not a leak."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    carol_token = await _register_login(client, _CAROL, key_fill=0x03)
    created = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    conversation_id = created.json()["id"]
    member = await client.get(
        f"/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert member.status_code == 200
    stranger = await client.get(
        f"/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {carol_token}"},
    )
    assert stranger.status_code == 404


# Confirm both epoch paths return the non-secret integer 0 for a fresh conversation.
async def test_epoch_endpoints_return_current_counter(client: AsyncClient) -> None:
    """Read epoch from REST and from the spec §6.4 /keys alias."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    created = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    conversation_id = created.json()["id"]
    rest = await client.get(
        f"/conversations/{conversation_id}/epoch",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    alias = await client.get(
        f"/keys/conversations/{conversation_id}/epoch",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert rest.status_code == 200
    assert alias.status_code == 200
    assert rest.json()["current_epoch"] == 0
    assert alias.json()["current_epoch"] == 0
    assert rest.json()["conversation_id"] == conversation_id


# Confirm epoch read is authenticated and membership-gated.
async def test_epoch_requires_auth_and_membership(client: AsyncClient) -> None:
    """Reject a missing token and a non-member with 401/403 and 404 respectively."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    carol_token = await _register_login(client, _CAROL, key_fill=0x03)
    created = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    conversation_id = created.json()["id"]
    unauthenticated = await client.get(f"/conversations/{conversation_id}/epoch")
    assert unauthenticated.status_code in (401, 403)
    stranger = await client.get(
        f"/keys/conversations/{conversation_id}/epoch",
        headers={"Authorization": f"Bearer {carol_token}"},
    )
    assert stranger.status_code == 404


# Confirm message reads are scoped by conversation_id and cannot see another pair's rows.
async def test_message_queries_are_scoped_by_conversation_id(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Persist envelopes in two conversations and require each listing to stay isolated."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    bob_token = await _register_login(client, _BOB, key_fill=0x02)
    await _register_login(client, _CAROL, key_fill=0x03)
    alice_bob = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    alice_carol = await client.post(
        "/conversations",
        json={"peer_username": "carol"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    alice_bob_id = UUID(alice_bob.json()["id"])
    alice_carol_id = UUID(alice_carol.json()["id"])
    alice = await db_session.scalar(select(User).where(User.username == "alice"))
    assert alice is not None

    await persist_envelope(
        db_session,
        conversation_id=alice_bob_id,
        sender_id=alice.id,
        ciphertext=b"ciphertext-for-bob-only",
        nonce=b"n" * 24,
        key_epoch=0,
    )
    await persist_envelope(
        db_session,
        conversation_id=alice_carol_id,
        sender_id=alice.id,
        ciphertext=b"ciphertext-for-carol-only",
        nonce=b"n" * 24,
        key_epoch=0,
    )

    bob_rows = await list_envelopes_for_conversation(db_session, alice_bob_id)
    carol_rows = await list_envelopes_for_conversation(db_session, alice_carol_id)
    assert [row.ciphertext for row in bob_rows] == [b"ciphertext-for-bob-only"]
    assert [row.ciphertext for row in carol_rows] == [b"ciphertext-for-carol-only"]
    # The unused bob_token documents that Bob is a real second account, not a dummy username.
    assert bob_token
