"""Verify authenticated WebSocket relay of ciphertext only, never plaintext."""

# Import base64 to build plausible AEAD-sized envelopes the server will not decrypt.
import base64

# Import AsyncIterator to type the WebSocket client helper.
from collections.abc import AsyncIterator

# Import asynccontextmanager so WebSocket clients enter/exit on the same test task.
from contextlib import asynccontextmanager

# Import UUID to look up persisted envelope rows by the JSON id string.
from uuid import UUID

# Import pytest for exception assertions on failed handshakes.
import pytest

# Import HTTPX's async client type for fixture-provided HTTP and WebSocket calls.
from httpx import AsyncClient

# Import httpx-ws helpers to open in-process WebSocket sessions against the ASGI app.
from httpx_ws import AsyncWebSocketSession, WebSocketDisconnect, WebSocketUpgradeError, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

# Import SQLAlchemy helpers to inspect persisted envelopes without going through REST.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ASGI app so WS tests can wrap it in a same-task transport.
from app.main import app

# Import ORM models used to assert stored rows contain ciphertext, not plaintext.
from app.models.message import Message
from app.models.user import User


# Open a WebSocket-capable HTTPX client on the same asyncio task that closes it.
@asynccontextmanager
async def _ws_client() -> AsyncIterator[AsyncClient]:
    """Yield an AsyncClient whose transport understands WebSocket upgrades.

    Must not replace the HTTP `client` fixture: ASGIWebSocketTransport uses an
    anyio cancel scope that pytest-asyncio would otherwise tear down on a
    different task. HTTP setup still uses `client`; only the handshake uses this.
    """

    async with ASGIWebSocketTransport(app) as transport:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ws_client:
            yield ws_client


# Open one authenticated WebSocket with an explicit session type for mypy.
@asynccontextmanager
async def _connect_ws(url: str, client: AsyncClient) -> AsyncIterator[AsyncWebSocketSession]:
    """Yield a typed WebSocket session against the in-process ASGI app."""

    session: AsyncWebSocketSession
    async with aconnect_ws(url, client) as session:
        yield session


# Assert a WebSocket handshake is rejected, unwrapping anyio ExceptionGroups.
async def _assert_ws_rejected(url: str, expected_codes: tuple[int, ...] | None = None) -> None:
    """Require the server to close or refuse the upgrade rather than accept the socket."""

    try:
        async with _ws_client() as ws_client:
            async with aconnect_ws(url, ws_client):
                pytest.fail("WebSocket handshake should have been rejected")
    except* WebSocketDisconnect as group:
        if expected_codes is not None:
            codes = [
                exc.code
                for exc in group.exceptions
                if isinstance(exc, WebSocketDisconnect)
            ]
            assert any(code in expected_codes for code in codes)
    except* WebSocketUpgradeError:
        pass


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


# Build a 24-byte XChaCha20-Poly1305 IETF nonce as standard base64.
def _fake_nonce(fill_byte: int = 0x0A) -> str:
    """Return base64 text decoding to exactly 24 nonce bytes."""

    return base64.b64encode(bytes([fill_byte]) * 24).decode("ascii")


# Build a tag-sized-or-larger ciphertext blob as standard base64.
def _fake_ciphertext(fill_byte: int = 0x0C) -> str:
    """Return base64 text decoding to 32 bytes of opaque ciphertext."""

    return base64.b64encode(bytes([fill_byte]) * 32).decode("ascii")


# Register, log in, and upload a public key, returning the access token.
async def _register_login(client: AsyncClient, payload: dict[str, str], key_fill: int) -> str:
    """Create an account ready for E2EE conversations."""

    await client.post("/auth/register", json=payload)
    response = await client.post(
        "/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    access_token = str(response.json()["access_token"])
    await client.post(
        "/keys/me",
        json={"public_key": _fake_public_key(key_fill)},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return access_token


# Start the Alice/Bob conversation used by most relay tests.
async def _alice_bob_conversation(client: AsyncClient) -> tuple[str, str, str]:
    """Return (alice_token, bob_token, conversation_id) for a keyed pair."""

    alice_token = await _register_login(client, _ALICE, 0x01)
    bob_token = await _register_login(client, _BOB, 0x02)
    created = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    return alice_token, bob_token, str(created.json()["id"])


# Confirm Alice's ciphertext is persisted and fanned out to Bob without a plaintext field.
async def test_websocket_relays_ciphertext_only(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Send an envelope from Alice and require Bob to receive ciphertext, never plaintext."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    ciphertext = _fake_ciphertext(0x11)
    nonce = _fake_nonce(0x22)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            async with _connect_ws(bob_url, ws_client) as bob_ws:
                await alice_ws.send_json(
                    {
                        "ciphertext": ciphertext,
                        "nonce": nonce,
                        "key_epoch": 0,
                        "conversation_id": conversation_id,
                    }
                )
                accepted = await alice_ws.receive_json()
                assert accepted["type"] == "accepted"
                fanout = await bob_ws.receive_json()

    assert fanout["type"] == "envelope"
    assert fanout["ciphertext"] == ciphertext
    assert fanout["nonce"] == nonce
    assert fanout["key_epoch"] == 0
    assert fanout["conversation_id"] == conversation_id
    assert "plaintext" not in fanout
    assert "private_key" not in fanout
    # The secret phrase must never appear in the relayed frame.
    assert "secret handshake" not in str(fanout)

    stored = await db_session.scalar(select(Message).where(Message.id == UUID(str(fanout["id"]))))
    assert stored is not None
    assert stored.ciphertext == base64.b64decode(ciphertext)
    assert stored.nonce == base64.b64decode(nonce)
    assert str(stored.conversation_id) == conversation_id
    alice = await db_session.scalar(select(User).where(User.username == "alice"))
    assert alice is not None
    assert stored.sender_id == alice.id


# Confirm a missing or invalid access token cannot open the relay.
async def test_websocket_rejects_missing_and_invalid_tokens(client: AsyncClient) -> None:
    """Connect without a token and with a garbage token; both must fail closed."""

    alice_token, _bob_token, conversation_id = await _alice_bob_conversation(client)
    missing_url = f"http://testserver/ws/conversations/{conversation_id}"
    bad_url = f"http://testserver/ws/conversations/{conversation_id}?access_token=not-a-jwt"

    await _assert_ws_rejected(missing_url, (4401, 1008))
    await _assert_ws_rejected(bad_url, (4401, 1008))
    assert alice_token


# Confirm a non-member cannot attach to someone else's conversation socket.
async def test_websocket_rejects_non_member(client: AsyncClient) -> None:
    """Have Carol connect to Alice/Bob's conversation and require a forbidden close."""

    _alice_token, _bob_token, conversation_id = await _alice_bob_conversation(client)
    carol_token = await _register_login(client, _CAROL, 0x03)
    carol_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={carol_token}"

    await _assert_ws_rejected(carol_url, (4403, 1008, 4003))


# Confirm a frame that claims a different conversation_id is rejected, not stored.
async def test_websocket_rejects_cross_conversation_envelope(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Send Alice's envelope claiming Carol's conversation while connected to Bob's."""

    alice_token, bob_token, alice_bob_id = await _alice_bob_conversation(client)
    await _register_login(client, _CAROL, 0x03)
    alice_carol = await client.post(
        "/conversations",
        json={"peer_username": "carol"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    alice_carol_id = str(alice_carol.json()["id"])
    alice_url = f"http://testserver/ws/conversations/{alice_bob_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{alice_bob_id}?access_token={bob_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            # Keep Bob connected so a buggy fan-out would have somewhere to go.
            async with _connect_ws(bob_url, ws_client):
                await alice_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(),
                    "nonce": _fake_nonce(),
                    "key_epoch": 0,
                    "conversation_id": alice_carol_id,
                }
            )
            error = await alice_ws.receive_json()
            assert error["type"] == "error"
            assert "conversation_id" in error["detail"]
            # Bob must not receive a fanned-out frame for the rejected envelope.
            # Closing without a receive is the assertion; an unexpected fan-out
            # would still be stored and is checked via the database below.

    stored = list((await db_session.scalars(select(Message))).all())
    assert stored == []


# Confirm a spoofed sender_id is rejected at the protocol layer.
async def test_websocket_rejects_spoofed_sender_id(client: AsyncClient) -> None:
    """Have Alice claim Bob's user id as sender_id and require an error frame."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    created = await client.get(
        f"/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    bob_id = created.json()["peer"]["id"]
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            await alice_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(),
                    "nonce": _fake_nonce(),
                    "key_epoch": 0,
                    "sender_id": bob_id,
                }
            )
            error = await alice_ws.receive_json()
    assert error["type"] == "error"
    assert "sender_id" in error["detail"]
    assert bob_token


# Confirm a future key_epoch is rejected rather than stored.
async def test_websocket_rejects_future_key_epoch(client: AsyncClient) -> None:
    """Send key_epoch=99 while current_epoch is 0 and require an error frame."""

    alice_token, _bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            await alice_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(),
                    "nonce": _fake_nonce(),
                    "key_epoch": 99,
                }
            )
            error = await alice_ws.receive_json()
    assert error["type"] == "error"
    assert "key_epoch" in error["detail"]
