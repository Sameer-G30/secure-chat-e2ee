"""Verify safe message editing (v2 associated data), delete-for-everyone, and delete-for-me."""

# Import base64 to build plausible AEAD-sized envelopes and public keys.
import base64

# Import AsyncIterator to type the WebSocket client helper.
from collections.abc import AsyncIterator

# Import asynccontextmanager so WebSocket clients enter/exit on the same test task.
from contextlib import asynccontextmanager

# Import uuid4 to generate client-chosen message identities, matching a real client.
from uuid import uuid4

# Import HTTPX's async client type for fixture-provided HTTP and WebSocket calls.
from httpx import AsyncClient

# Import httpx-ws helpers to open in-process WebSocket sessions against the ASGI app.
from httpx_ws import AsyncWebSocketSession, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

# Import SQLAlchemy helpers to inspect persisted rows without going through REST.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ASGI app so WS tests can wrap it in a same-task transport.
from app.main import app

# Import the Message model to assert stored rows after edits/deletes.
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


# Confirm a v2 send-then-edit updates the same row in place with an advanced revision.
async def test_editing_a_v2_message_updates_the_same_row(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Send an original v2 envelope, then an edit, and require one updated row."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"
    message_id = str(uuid4())

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            async with _connect_ws(bob_url, ws_client) as bob_ws:
                # Original send: ad_version=2 because message_id is present.
                await alice_ws.send_json(
                    {
                        "ciphertext": _fake_ciphertext(0x11),
                        "nonce": _fake_nonce(0x22),
                        "key_epoch": 0,
                        "message_id": message_id,
                        "revision": 0,
                    }
                )
                accepted = await _receive_until(alice_ws, "accepted")
                original_row_id = accepted["id"]
                original = await _receive_until(bob_ws, "envelope")
                assert original["message_id"] == message_id
                assert original["revision"] == 0
                assert original["ad_version"] == 2
                assert original["edited_at"] is None

                # Edit: same message_id, revision advances by exactly one.
                await alice_ws.send_json(
                    {
                        "ciphertext": _fake_ciphertext(0x33),
                        "nonce": _fake_nonce(0x44),
                        "key_epoch": 0,
                        "message_id": message_id,
                        "revision": 1,
                    }
                )
                edit_accepted = await _receive_until(alice_ws, "accepted")
                # An edit updates the existing row; the id never changes.
                assert edit_accepted["id"] == original_row_id
                edited = await _receive_until(bob_ws, "envelope")

    assert edited["message_id"] == message_id
    assert edited["revision"] == 1
    assert edited["ciphertext"] == _fake_ciphertext(0x33)
    assert edited["edited_at"] is not None

    stored = list((await db_session.scalars(select(Message))).all())
    # One row, not two: the edit overwrote it in place.
    assert len(stored) == 1
    assert stored[0].revision == 1
    assert stored[0].ciphertext == base64.b64decode(_fake_ciphertext(0x33))
    assert stored[0].edited_at is not None


# Confirm a revision that does not advance by exactly one is rejected, not applied.
async def test_editing_with_wrong_revision_is_rejected(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Skip straight to revision=2 and require an error frame plus an unchanged row."""

    alice_token, _bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    message_id = str(uuid4())

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            await alice_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(0x11),
                    "nonce": _fake_nonce(0x22),
                    "key_epoch": 0,
                    "message_id": message_id,
                    "revision": 0,
                }
            )
            await _receive_until(alice_ws, "accepted")

            await alice_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(0x99),
                    "nonce": _fake_nonce(0x88),
                    "key_epoch": 0,
                    "message_id": message_id,
                    "revision": 2,
                }
            )
            error = await _receive_until(alice_ws, "error")

    assert error["type"] == "error"
    assert "revision" in str(error["detail"])
    stored = list((await db_session.scalars(select(Message))).all())
    assert len(stored) == 1
    assert stored[0].revision == 0


# Confirm one sender cannot "edit" another sender's message by reusing their message_id.
async def test_resending_someone_elses_message_id_creates_a_new_row_not_an_edit(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Have Bob reuse Alice's message_id and require a second, independent row."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"
    message_id = str(uuid4())

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            await alice_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(0x11),
                    "nonce": _fake_nonce(0x22),
                    "key_epoch": 0,
                    "message_id": message_id,
                    "revision": 0,
                }
            )
            alice_accepted = await _receive_until(alice_ws, "accepted")

        async with _connect_ws(bob_url, ws_client) as bob_ws:
            await bob_ws.send_json(
                {
                    "ciphertext": _fake_ciphertext(0x55),
                    "nonce": _fake_nonce(0x66),
                    "key_epoch": 0,
                    "message_id": message_id,
                    "revision": 1,
                }
            )
            bob_accepted = await _receive_until(bob_ws, "accepted")

    # Two distinct rows: Bob's resend never matched Alice's row (sender_id differs).
    assert alice_accepted["id"] != bob_accepted["id"]
    stored = list((await db_session.scalars(select(Message))).all())
    assert len(stored) == 2


# Confirm delete-for-everyone is a hard delete and the peer is notified.
async def test_delete_for_everyone_removes_the_row_and_notifies_the_peer(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Alice deletes her own message and Bob receives a message_deleted frame."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            async with _connect_ws(bob_url, ws_client) as bob_ws:
                await alice_ws.send_json(
                    {
                        "ciphertext": _fake_ciphertext(),
                        "nonce": _fake_nonce(),
                        "key_epoch": 0,
                    }
                )
                accepted = await _receive_until(alice_ws, "accepted")
                await _receive_until(bob_ws, "envelope")
                message_id = accepted["id"]

                deleted = await client.delete(
                    f"/conversations/{conversation_id}/messages/{message_id}",
                    headers={"Authorization": f"Bearer {alice_token}"},
                )
                assert deleted.status_code == 204

                deletion_frame = await _receive_until(bob_ws, "message_deleted")

    assert deletion_frame["id"] == message_id
    assert deletion_frame["conversation_id"] == conversation_id
    stored = list((await db_session.scalars(select(Message))).all())
    assert stored == []
    assert bob_token


# Confirm only the sender may delete-for-everyone; a non-sender gets 403, not silent success.
async def test_delete_for_everyone_by_non_sender_is_forbidden(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Have Bob try to delete Alice's message and require a 403, row still present."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            await alice_ws.send_json(
                {"ciphertext": _fake_ciphertext(), "nonce": _fake_nonce(), "key_epoch": 0}
            )
            accepted = await _receive_until(alice_ws, "accepted")
            message_id = accepted["id"]

    response = await client.delete(
        f"/conversations/{conversation_id}/messages/{message_id}",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert response.status_code == 403
    stored = list((await db_session.scalars(select(Message))).all())
    assert len(stored) == 1


# Confirm deleting a message id that does not exist 404s.
async def test_delete_unknown_message_returns_404(client: AsyncClient) -> None:
    """Delete a random UUID that was never a stored message."""

    alice_token, _bob_token, conversation_id = await _alice_bob_conversation(client)
    response = await client.delete(
        f"/conversations/{conversation_id}/messages/{uuid4()}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 404


# Confirm delete-for-me only affects the hiding owner's own history, not the peer's.
async def test_hide_for_me_only_affects_the_hiding_owners_history(client: AsyncClient) -> None:
    """Bob hides Alice's message; it disappears from Bob's history, stays in Alice's."""

    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            await alice_ws.send_json(
                {"ciphertext": _fake_ciphertext(), "nonce": _fake_nonce(), "key_epoch": 0}
            )
            accepted = await _receive_until(alice_ws, "accepted")
    message_id = accepted["id"]

    hidden = await client.post(
        f"/conversations/{conversation_id}/messages/{message_id}/hide",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert hidden.status_code == 204

    bob_history = await client.get(
        f"/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    alice_history = await client.get(
        f"/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert bob_history.json()["messages"] == []
    assert len(alice_history.json()["messages"]) == 1


# Confirm hiding a message id outside the conversation 404s rather than leaking existence.
async def test_hide_message_outside_conversation_returns_404(client: AsyncClient) -> None:
    """Hide a random UUID that is not a message in this conversation."""

    alice_token, _bob_token, conversation_id = await _alice_bob_conversation(client)
    response = await client.post(
        f"/conversations/{conversation_id}/messages/{uuid4()}/hide",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 404
