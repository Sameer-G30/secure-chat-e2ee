"""Verify receipts, unread badges, public profiles, and sealed image blobs."""

# Import base64 to build well-formed public-key and blob payloads for test accounts.
import base64

# Import UUID to inspect conversation and envelope identifiers.
from uuid import UUID

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import SQLAlchemy helpers to inspect stored rows.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import ORM models used for direct row assertions.
from app.models.message_receipt import MessageReceipt
from app.models.user import User

# Import the conversation-scoped envelope writer used to seed unread tests.
from app.services.relay import persist_envelope

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


# Build a syntactically valid base64 X25519-sized (32-byte) public key for test accounts.
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


# Confirm GET /contacts reports unread inbound envelopes until mark-read.
async def test_unread_count_clears_after_mark_read(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Seed one Bob->Alice envelope and require unread_count 1, then 0 after POST /read."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    await client.post(
        "/contacts",
        json={"username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    started = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert started.status_code == 200
    conversation_id = UUID(started.json()["id"])
    bob = await db_session.scalar(select(User).where(User.username == "bob"))
    assert bob is not None
    await persist_envelope(
        db_session,
        conversation_id=conversation_id,
        sender_id=bob.id,
        ciphertext=bytes([0x0C]) * 32,
        nonce=bytes([0x0A]) * 24,
        key_epoch=0,
    )
    listed = await client.get("/contacts", headers={"Authorization": f"Bearer {alice_token}"})
    assert listed.status_code == 200
    bob_row = next(row for row in listed.json()["contacts"] if row["username"] == "bob")
    assert bob_row["unread_count"] == 1
    marked = await client.post(
        f"/conversations/{conversation_id}/read",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert marked.status_code == 200
    listed_again = await client.get("/contacts", headers={"Authorization": f"Bearer {alice_token}"})
    bob_row_again = next(
        row for row in listed_again.json()["contacts"] if row["username"] == "bob"
    )
    assert bob_row_again["unread_count"] == 0


# Confirm history attaches peer_read after the recipient marks the conversation read.
async def test_history_includes_peer_read_for_own_sent_envelopes(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Alice's history should show peer_read once Bob POSTs /read."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    bob_token = await _register_login(client, _BOB, key_fill=0x02)
    started = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    conversation_id = UUID(started.json()["id"])
    alice = await db_session.scalar(select(User).where(User.username == "alice"))
    assert alice is not None
    stored = await persist_envelope(
        db_session,
        conversation_id=conversation_id,
        sender_id=alice.id,
        ciphertext=bytes([0x0C]) * 32,
        nonce=bytes([0x0A]) * 24,
        key_epoch=0,
    )
    before = await client.get(
        f"/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert before.status_code == 200
    own = next(row for row in before.json()["messages"] if row["id"] == str(stored.id))
    assert own["peer_delivered"] is False
    assert own["peer_read"] is False
    marked = await client.post(
        f"/conversations/{conversation_id}/read",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert marked.status_code == 200
    after = await client.get(
        f"/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    own_after = next(row for row in after.json()["messages"] if row["id"] == str(stored.id))
    assert own_after["peer_delivered"] is True
    assert own_after["peer_read"] is True
    receipt = await db_session.scalar(
        select(MessageReceipt).where(MessageReceipt.message_id == stored.id)
    )
    assert receipt is not None
    assert receipt.read_at is not None


# Confirm PATCH /users/me updates display name and bio without exposing a message body.
async def test_patch_me_updates_public_profile(client: AsyncClient) -> None:
    """Alice can set a display name and bio, then Bob can read them on /profile."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    bob_token = await _register_login(client, _BOB, key_fill=0x02)
    patched = await client.patch(
        "/users/me",
        json={"display_name": "Alice A.", "bio": "Ciphertext only."},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert patched.status_code == 200
    assert patched.json()["display_name"] == "Alice A."
    assert patched.json()["bio"] == "Ciphertext only."
    assert patched.json()["email"] == "alice@example.com"
    public = await client.get(
        "/users/alice/profile",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert public.status_code == 200
    assert public.json()["username"] == "alice"
    assert public.json()["display_name"] == "Alice A."
    assert public.json()["bio"] == "Ciphertext only."
    assert public.json()["has_avatar"] is False
    assert "email" not in public.json()


# Confirm avatar upload is bounded and served only to authenticated callers.
async def test_avatar_upload_and_authenticated_fetch(client: AsyncClient) -> None:
    """POST a tiny JPEG-typed avatar, then GET it with a bearer token."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    bob_token = await _register_login(client, _BOB, key_fill=0x02)
    # A 1x1 JPEG is unnecessary; the endpoint checks media type and size, not pixels.
    tiny = b"\xff\xd8\xff\xd9" + b"\x00" * 16
    uploaded = await client.post(
        "/users/me/avatar",
        files={"file": ("avatar.jpg", tiny, "image/jpeg")},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["has_avatar"] is True
    fetched = await client.get(
        "/users/alice/avatar",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert fetched.status_code == 200
    assert fetched.headers["content-type"].startswith("image/jpeg")
    assert fetched.content == tiny
    anonymous = await client.get("/users/alice/avatar")
    assert anonymous.status_code == 401


# Confirm sealed blobs are membership-gated and never opened by the server.
async def test_encrypted_blob_round_trip_is_membership_scoped(client: AsyncClient) -> None:
    """Alice can store and reload opaque bytes; Carol cannot."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    carol_token = await _register_login(client, _CAROL, key_fill=0x03)
    started = await client.post(
        "/conversations",
        json={"peer_username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    conversation_id = started.json()["id"]
    blob_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    ciphertext = base64.b64encode(bytes([0x0C]) * 32).decode("ascii")
    nonce = base64.b64encode(bytes([0x0A]) * 24).decode("ascii")
    created = await client.post(
        f"/conversations/{conversation_id}/blobs",
        json={"id": blob_id, "ciphertext": ciphertext, "nonce": nonce},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert created.status_code == 201
    assert created.json()["id"] == blob_id
    assert created.json()["ciphertext"] == ciphertext
    loaded = await client.get(
        f"/conversations/{conversation_id}/blobs/{blob_id}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert loaded.status_code == 200
    assert loaded.json()["nonce"] == nonce
    forbidden = await client.get(
        f"/conversations/{conversation_id}/blobs/{blob_id}",
        headers={"Authorization": f"Bearer {carol_token}"},
    )
    assert forbidden.status_code == 404
    duplicate = await client.post(
        f"/conversations/{conversation_id}/blobs",
        json={"id": blob_id, "ciphertext": ciphertext, "nonce": nonce},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert duplicate.status_code == 409


# Confirm contacts include profile flags used by the sidebar avatar.
async def test_contacts_include_display_name_and_unread_defaults(client: AsyncClient) -> None:
    """A newly added contact has unread_count 0 and has_avatar false."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    created = await client.post(
        "/contacts",
        json={"username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert created.status_code == 200
    assert created.json()["unread_count"] == 0
    assert created.json()["has_avatar"] is False
    assert created.json()["display_name"] is None
