"""Verify server-enforced blocking, unlike the legacy app's localStorage-only feature."""

# Import base64 to build plausible AEAD-sized envelopes and public keys.
import base64

# Import AsyncIterator to type the WebSocket client helper.
from collections.abc import AsyncIterator

# Import asynccontextmanager so WebSocket clients enter/exit on the same test task.
from contextlib import asynccontextmanager

# Import HTTPX's async client type for fixture-provided HTTP and WebSocket calls.
from httpx import AsyncClient

# Import httpx-ws helpers to open in-process WebSocket sessions against the ASGI app.
from httpx_ws import AsyncWebSocketSession, WebSocketDisconnect, WebSocketUpgradeError, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

# Import SQLAlchemy helpers to inspect persisted rows without going through REST.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ASGI app so WS tests can wrap it in a same-task transport.
from app.main import app

# Import the Message model to assert a dropped envelope is never persisted.
from app.models.message import Message

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


def _fake_public_key(fill_byte: int = 0x01) -> str:
    """Return base64 text decoding to exactly 32 bytes, as a real key would."""

    return base64.b64encode(bytes([fill_byte]) * 32).decode("ascii")


def _fake_nonce(fill_byte: int = 0x0A) -> str:
    """Return base64 text decoding to exactly 24 nonce bytes."""

    return base64.b64encode(bytes([fill_byte]) * 24).decode("ascii")


def _fake_ciphertext(fill_byte: int = 0x0C) -> str:
    """Return base64 text decoding to 32 bytes of opaque ciphertext."""

    return base64.b64encode(bytes([fill_byte]) * 32).decode("ascii")


async def _register_login(client: AsyncClient, payload: dict[str, str], key_fill: int) -> str:
    """Create an account, log in, and upload a public key."""

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


@asynccontextmanager
async def _ws_client() -> AsyncIterator[AsyncClient]:
    """Yield an AsyncClient whose transport understands WebSocket upgrades."""

    async with ASGIWebSocketTransport(app) as transport:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ws_client:
            yield ws_client


@asynccontextmanager
async def _connect_ws(url: str, client: AsyncClient) -> AsyncIterator[AsyncWebSocketSession]:
    """Yield a typed WebSocket session against the in-process ASGI app."""

    session: AsyncWebSocketSession
    async with aconnect_ws(url, client) as session:
        yield session


async def _receive_until(ws: AsyncWebSocketSession, expected_type: str) -> dict[str, object]:
    """Drain presence/typing/epoch frames, then return the next frame of the requested type."""

    while True:
        frame = await ws.receive_json()
        assert isinstance(frame, dict)
        if frame.get("type") in {"presence", "typing", "epoch"} and expected_type not in {
            "presence",
            "typing",
            "epoch",
        }:
            continue
        if expected_type != "any" and frame.get("type") != expected_type:
            continue
        return frame


async def _assert_ws_rejected(url: str, expected_codes: tuple[int, ...]) -> None:
    """Require the server to close or refuse the upgrade rather than accept the socket."""

    try:
        async with _ws_client() as ws_client:
            async with aconnect_ws(url, ws_client):
                raise AssertionError("WebSocket handshake should have been rejected")
    except* WebSocketDisconnect as group:
        codes = [exc.code for exc in group.exceptions if isinstance(exc, WebSocketDisconnect)]
        assert any(code in expected_codes for code in codes)
    except* WebSocketUpgradeError:
        pass


# Confirm blocking, listing, and unblocking round-trip correctly.
async def test_block_list_and_unblock_round_trip(client: AsyncClient) -> None:
    """Block Bob, see him in the list, unblock him, and see an empty list."""

    alice_token = await _register_login(client, _ALICE, 0x01)
    await _register_login(client, _BOB, 0x02)
    headers = {"Authorization": f"Bearer {alice_token}"}

    blocked = await client.post("/blocks", json={"username": "bob"}, headers=headers)
    assert blocked.status_code == 200
    assert blocked.json()["username"] == "bob"

    listed = await client.get("/blocks", headers=headers)
    assert [row["username"] for row in listed.json()["blocks"]] == ["bob"]

    unblocked = await client.delete("/blocks/bob", headers=headers)
    assert unblocked.status_code == 204

    listed_again = await client.get("/blocks", headers=headers)
    assert listed_again.json()["blocks"] == []


# Confirm blocking the same account twice is idempotent, not a second row.
async def test_block_is_idempotent(client: AsyncClient) -> None:
    """Block Bob twice and require a single list entry."""

    alice_token = await _register_login(client, _ALICE, 0x01)
    await _register_login(client, _BOB, 0x02)
    headers = {"Authorization": f"Bearer {alice_token}"}

    first = await client.post("/blocks", json={"username": "bob"}, headers=headers)
    second = await client.post("/blocks", json={"username": "bob"}, headers=headers)
    assert first.json()["id"] == second.json()["id"]
    listed = await client.get("/blocks", headers=headers)
    assert len(listed.json()["blocks"]) == 1


# Confirm an account cannot block itself.
async def test_cannot_block_self(client: AsyncClient) -> None:
    """Ask Alice to block alice and require a 400."""

    alice_token = await _register_login(client, _ALICE, 0x01)
    response = await client.post(
        "/blocks",
        json={"username": "alice"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 400


# Confirm blocking an unknown username is not a user-existence oracle beyond 404.
async def test_block_unknown_user_returns_404(client: AsyncClient) -> None:
    """Ask Alice to block a handle that was never registered."""

    alice_token = await _register_login(client, _ALICE, 0x01)
    response = await client.post(
        "/blocks",
        json={"username": "nobody"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 404


# Confirm unblocking an account that was never blocked is a no-op, not an error.
async def test_unblock_nonexistent_block_is_a_noop(client: AsyncClient) -> None:
    """Unblock Bob without ever having blocked him and require 204."""

    alice_token = await _register_login(client, _ALICE, 0x01)
    await _register_login(client, _BOB, 0x02)
    response = await client.delete(
        "/blocks/bob", headers={"Authorization": f"Bearer {alice_token}"}
    )
    assert response.status_code == 204


# Confirm block REST requires a bearer access token.
async def test_blocks_require_authentication(client: AsyncClient) -> None:
    """GET/POST/DELETE /blocks with no Authorization header and require 401 or 403."""

    listed = await client.get("/blocks")
    added = await client.post("/blocks", json={"username": "bob"})
    removed = await client.delete("/blocks/bob")
    assert listed.status_code in (401, 403)
    assert added.status_code in (401, 403)
    assert removed.status_code in (401, 403)


# Confirm a blocked pair's relay socket is refused outright, in either direction.
async def test_websocket_refuses_connection_when_either_side_has_blocked_the_other(
    client: AsyncClient,
) -> None:
    """Have Alice block Bob, then require both Alice's and Bob's sockets to be refused."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    await client.post(
        "/blocks",
        json={"username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"

    await _assert_ws_rejected(alice_url, (4403, 1008, 4003))
    await _assert_ws_rejected(bob_url, (4403, 1008, 4003))


# Confirm a message the sender managed to queue before a block still fake-acks
# the sender without ever persisting or delivering it (defense in depth beyond
# the WS-connect refusal above).
async def test_blocked_senders_envelope_is_silently_dropped_not_delivered(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Simulate a race: connect first, block second, then send and require a silent drop."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            # Bob blocks Alice mid-conversation, after Alice's socket is already open.
            await client.post(
                "/blocks",
                json={"username": "alice"},
                headers={"Authorization": f"Bearer {bob_token}"},
            )
            await alice_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(),
                    "nonce": _fake_nonce(),
                    "key_epoch": 0,
                }
            )
            # The sender still gets a fake "accepted" ack so their composer does not
            # show a stuck-sending state; the block's existence is never disclosed.
            accepted = await _receive_until(alice_ws, "accepted")
            assert accepted["type"] == "accepted"

    stored = list((await db_session.scalars(select(Message))).all())
    assert stored == []
