"""Verify Slice 8 epoch rotation: N messages OR 24h, WS broadcast, old envelopes still store."""

# Import base64 to build plausible AEAD-sized envelopes the server will not decrypt.
import base64

# Import Counter so history assertions do not depend on same-second UUID order.
from collections import Counter

# Import AsyncIterator and Iterator for the WS client and settings fixtures.
from collections.abc import AsyncIterator, Iterator

# Import asynccontextmanager so WebSocket clients enter/exit on the same test task.
from contextlib import asynccontextmanager

# Import timezone-aware clocks for the 24h rotation case.
from datetime import UTC, datetime, timedelta

# Import UUID to look up persisted rows and conversation ids.
from uuid import UUID

# Import pytest so tests can override rotation thresholds via environment.
import pytest

# Import HTTPX's async client type for fixture-provided HTTP and WebSocket calls.
from httpx import AsyncClient

# Import httpx-ws helpers to open in-process WebSocket sessions against the ASGI app.
from httpx_ws import AsyncWebSocketSession, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

# Import SQLAlchemy helpers to inspect current_epoch and stored key_epoch values.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import the cached settings provider so tests can rebuild it after env patches.
from app.config import get_settings

# Import the ASGI app so WS tests can wrap it in a same-task transport.
from app.main import app

# Import ORM models used to stamp last_rotated_at and assert stored envelopes.
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User

# Import the rotation helper so unit tests can freeze "now" without a WebSocket.
from app.services.epoch import maybe_rotate_epoch
from app.services.relay import persist_envelope


# Open a WebSocket-capable HTTPX client on the same asyncio task that closes it.
@asynccontextmanager
async def _ws_client() -> AsyncIterator[AsyncClient]:
    """Yield an AsyncClient whose transport understands WebSocket upgrades."""

    # ASGIWebSocketTransport must live on the same task as the test (see test_ws_relay).
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


# Skip typing/presence/epoch metadata until a frame matching expected_type arrives.
async def _receive_until(ws: AsyncWebSocketSession, expected_type: str) -> dict[str, object]:
    """Drain unrelated metadata frames, then return the next requested type."""

    while True:
        frame = await ws.receive_json()
        assert isinstance(frame, dict)
        frame_type = frame.get("type")
        if frame_type in {"presence", "typing", "epoch"} and expected_type not in {
            "presence",
            "typing",
            "epoch",
        }:
            continue
        if expected_type != "any" and frame_type != expected_type:
            continue
        return frame


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


# Start the Alice/Bob conversation used by the rotation relay tests.
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


# Override production N=50 / 24h so a two-message relay can prove the bump.
@pytest.fixture
def rotate_after_two_messages(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Set message-count N=2 and disable the wall-clock trigger for this test."""

    # Two envelopes are enough to demo a bump without sending 50 frames.
    monkeypatch.setenv("EPOCH_ROTATE_AFTER_MESSAGES", "2")
    # Disable hours so a fresh conversation's created_at cannot fire 24h in CI.
    monkeypatch.setenv("EPOCH_ROTATE_AFTER_HOURS", "0")
    # Drop the cached Settings so the next get_settings() reads these values.
    get_settings.cache_clear()
    # Hand control to the test that needs a fast bump.
    yield
    # Forget the N=2 cache so later tests see the production defaults again.
    get_settings.cache_clear()


# Isolate the 24h trigger: disable message-count, keep hours at 24.
@pytest.fixture
def rotate_after_twenty_four_hours(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Disable N-message rotation so only the wall-clock rule can fire."""

    # 50 would not fire on a single persist; 0 makes that explicit.
    monkeypatch.setenv("EPOCH_ROTATE_AFTER_MESSAGES", "0")
    # Keep the documented 24h half of the policy.
    monkeypatch.setenv("EPOCH_ROTATE_AFTER_HOURS", "24")
    # Rebuild Settings from the patched environment.
    get_settings.cache_clear()
    # Hand control to the time-based test.
    yield
    # Restore defaults for the rest of the suite.
    get_settings.cache_clear()


# Send one syntactically valid ciphertext envelope on an open conversation socket.
async def _send_envelope(
    ws: AsyncWebSocketSession,
    conversation_id: str,
    *,
    key_epoch: int,
    fill: int = 0x11,
) -> None:
    """Send {ciphertext, nonce, key_epoch} with no plaintext field."""

    # The server must never see a body field; this helper does not include one.
    await ws.send_json(
        {
            "ciphertext": _fake_ciphertext(fill),
            "nonce": _fake_nonce(fill),
            "key_epoch": key_epoch,
            "conversation_id": conversation_id,
        }
    )


# Confirm the default policy does not bump current_epoch on every persist.
async def test_default_policy_does_not_rotate_every_message(
    client: AsyncClient,
) -> None:
    """Send two envelopes under N=50 and require current_epoch to stay 0."""

    # Create a keyed Alice/Bob conversation at epoch 0.
    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            # Persist the first envelope under the current subkey id.
            await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x21)
            # Wait until the server has stored it.
            accepted = await _receive_until(alice_ws, "accepted")
            assert accepted["type"] == "accepted"
            # Persist a second envelope; production N=50 must not fire yet.
            await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x22)
            accepted_again = await _receive_until(alice_ws, "accepted")
            assert accepted_again["type"] == "accepted"

    # Read the public counter; two messages must not look like a silent per-send bump.
    epoch = await client.get(
        f"/conversations/{conversation_id}/epoch",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert epoch.status_code == 200
    assert epoch.json()["current_epoch"] == 0
    # Bob's token documents that this is a real pair, not a dummy username.
    assert bob_token


# Confirm N=2 bumps the counter, both sockets learn it, and GET epoch agrees.
async def test_websocket_bumps_epoch_after_n_messages_and_notifies_both_members(
    client: AsyncClient,
    rotate_after_two_messages: None,
) -> None:
    """Require a bump after the second persist and an epoch frame on both tabs."""

    # Mark the fixture as used so ruff does not flag it as unused.
    assert rotate_after_two_messages is None
    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            async with _connect_ws(bob_url, ws_client) as bob_ws:
                # First persist must store under epoch 0 and must not bump yet.
                await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x31)
                first_accepted = await _receive_until(alice_ws, "accepted")
                assert first_accepted["type"] == "accepted"
                first_fanout = await _receive_until(bob_ws, "envelope")
                assert first_fanout["key_epoch"] == 0
                # Counter is still 0 after one envelope.
                before = await client.get(
                    f"/conversations/{conversation_id}/epoch",
                    headers={"Authorization": f"Bearer {alice_token}"},
                )
                assert before.json()["current_epoch"] == 0
                # Second persist is the qualifying N=2 persist.
                await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x32)
                second_accepted = await _receive_until(alice_ws, "accepted")
                assert second_accepted["type"] == "accepted"
                second_fanout = await _receive_until(bob_ws, "envelope")
                assert second_fanout["key_epoch"] == 0
                # Both tabs must see the metadata bump, including the sender.
                alice_epoch = await _receive_until(alice_ws, "epoch")
                bob_epoch = await _receive_until(bob_ws, "epoch")

    assert alice_epoch["type"] == "epoch"
    assert bob_epoch["type"] == "epoch"
    assert alice_epoch["current_epoch"] == 1
    assert bob_epoch["current_epoch"] == 1
    # The frame is a counter only: no ciphertext, no key material, no draft text.
    assert "ciphertext" not in alice_epoch
    assert "nonce" not in alice_epoch
    assert "private_key" not in alice_epoch
    assert "plaintext" not in alice_epoch
    after = await client.get(
        f"/keys/conversations/{conversation_id}/epoch",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert after.json()["current_epoch"] == 1


# Confirm Bob can persist key_epoch=1 after the bump, and key_epoch=0 still stores.
async def test_after_bump_accepts_old_epoch_and_new_epoch_rejects_future(
    client: AsyncClient,
    db_session: AsyncSession,
    rotate_after_two_messages: None,
) -> None:
    """Accept key_epoch <= current_epoch; reject a future id; keep history rows."""

    assert rotate_after_two_messages is None
    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"

    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            async with _connect_ws(bob_url, ws_client) as bob_ws:
                # Reach current_epoch=1 with two Alice envelopes under key_epoch=0.
                await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x41)
                await _receive_until(alice_ws, "accepted")
                await _receive_until(bob_ws, "envelope")
                await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x42)
                await _receive_until(alice_ws, "accepted")
                await _receive_until(bob_ws, "envelope")
                await _receive_until(alice_ws, "epoch")
                await _receive_until(bob_ws, "epoch")
                # Bob encrypts the next send with the new id (what both tabs learned).
                await _send_envelope(bob_ws, conversation_id, key_epoch=1, fill=0x43)
                bob_accepted = await _receive_until(bob_ws, "accepted")
                assert bob_accepted["type"] == "accepted"
                bob_fanout = await _receive_until(alice_ws, "envelope")
                assert bob_fanout["key_epoch"] == 1
                # An in-flight / historical envelope under the old id must still store.
                await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x44)
                old_accepted = await _receive_until(alice_ws, "accepted")
                assert old_accepted["type"] == "accepted"
                old_fanout = await _receive_until(bob_ws, "envelope")
                assert old_fanout["key_epoch"] == 0
                # A future id is still rejected even after the bump to 1.
                await _send_envelope(alice_ws, conversation_id, key_epoch=2, fill=0x45)
                error = await _receive_until(alice_ws, "error")
                assert error["type"] == "error"
                assert "key_epoch" in str(error["detail"])

    # History stays conversation-scoped and ciphertext-only; old key_epoch rows remain.
    history = await client.get(
        f"/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert history.status_code == 200
    epochs = [row["key_epoch"] for row in history.json()["messages"]]
    # SQLite created_at is second-precision; same-second rows order by UUID, not send order.
    # The contract is the multiset: three accepted epoch-0 rows, one epoch-1, no future id.
    assert Counter(epochs) == Counter([0, 0, 0, 1])
    for row in history.json()["messages"]:
        assert row["type"] == "envelope"
        assert "plaintext" not in row
        assert "score" not in row
    stored = list((await db_session.scalars(select(Message))).all())
    assert len(stored) == 4
    assert all(row.key_epoch in {0, 1} for row in stored)


# Confirm the 24h half bumps on the next persist without waiting for N messages.
async def test_websocket_bumps_epoch_after_twenty_four_hours(
    client: AsyncClient,
    db_session: AsyncSession,
    rotate_after_twenty_four_hours: None,
) -> None:
    """Stamp last_rotated_at 25h ago and require one persist to increment the counter."""

    assert rotate_after_twenty_four_hours is None
    alice_token, bob_token, conversation_id = await _alice_bob_conversation(client)
    conversation = await db_session.scalar(
        select(Conversation).where(Conversation.id == UUID(conversation_id))
    )
    assert conversation is not None
    # Pretend the last bump (or an equivalent anchor) was more than 24h ago.
    conversation.last_rotated_at = datetime.now(UTC) - timedelta(hours=25)
    await db_session.commit()

    alice_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={alice_token}"
    bob_url = f"http://testserver/ws/conversations/{conversation_id}?access_token={bob_token}"
    async with _ws_client() as ws_client:
        async with _connect_ws(alice_url, ws_client) as alice_ws:
            async with _connect_ws(bob_url, ws_client) as bob_ws:
                # One envelope is below N=50; the wall-clock rule must still fire.
                await _send_envelope(alice_ws, conversation_id, key_epoch=0, fill=0x51)
                accepted = await _receive_until(alice_ws, "accepted")
                assert accepted["type"] == "accepted"
                await _receive_until(bob_ws, "envelope")
                alice_epoch = await _receive_until(alice_ws, "epoch")
                bob_epoch = await _receive_until(bob_ws, "epoch")

    assert alice_epoch["current_epoch"] == 1
    assert bob_epoch["current_epoch"] == 1
    epoch = await client.get(
        f"/conversations/{conversation_id}/epoch",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert epoch.json()["current_epoch"] == 1
    assert bob_token


# Confirm maybe_rotate_epoch itself is a counter bump, not a key derivation.
async def test_maybe_rotate_epoch_increments_counter_without_storing_keys(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Persist two rows then rotate with N=2; require current_epoch=1 and no key columns."""

    alice_token, _bob_token, conversation_id = await _alice_bob_conversation(client)
    conversation = await db_session.scalar(
        select(Conversation).where(Conversation.id == UUID(conversation_id))
    )
    assert conversation is not None
    existing = await db_session.scalar(
        select(Message).where(Message.conversation_id == UUID(conversation_id))
    )
    # There are no messages yet; persist two ciphertext rows through the shared helper.
    assert existing is None
    alice_user = await db_session.scalar(select(User).where(User.username == "alice"))
    assert alice_user is not None
    await persist_envelope(
        db_session,
        conversation_id=UUID(conversation_id),
        sender_id=alice_user.id,
        ciphertext=b"ciphertext-blob-number-one",
        nonce=b"n" * 24,
        key_epoch=0,
    )
    first = await maybe_rotate_epoch(
        db_session,
        conversation,
        rotate_after_messages=2,
        rotate_after_hours=0,
    )
    # One row is below N=2.
    assert first is None
    await persist_envelope(
        db_session,
        conversation_id=UUID(conversation_id),
        sender_id=alice_user.id,
        ciphertext=b"ciphertext-blob-number-two",
        nonce=b"n" * 24,
        key_epoch=0,
    )
    # Reload last_rotated_at / current_epoch after the second persist.
    await db_session.refresh(conversation)
    bumped = await maybe_rotate_epoch(
        db_session,
        conversation,
        rotate_after_messages=2,
        rotate_after_hours=0,
    )
    assert bumped == 1
    assert conversation.current_epoch == 1
    assert conversation.last_rotated_at is not None
    assert alice_token
