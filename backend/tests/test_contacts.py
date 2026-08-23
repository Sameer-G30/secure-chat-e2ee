"""Verify server-side contacts: add, list, idempotency, and self/unknown rejection."""

# Import base64 to build well-formed public-key payloads for test accounts.
import base64

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import SQLAlchemy helpers to inspect stored contact rows.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import ORM models used for direct row assertions.
from app.models.contact import Contact
from app.models.user import User

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


# Confirm Alice can save Bob and list him after a later GET.
async def test_add_and_list_contacts(client: AsyncClient, db_session: AsyncSession) -> None:
    """POST /contacts then GET /contacts and require Bob to appear by username."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    created = await client.post(
        "/contacts",
        json={"username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["username"] == "bob"
    assert "id" in body
    listed = await client.get("/contacts", headers={"Authorization": f"Bearer {alice_token}"})
    assert listed.status_code == 200
    contacts = listed.json()["contacts"]
    assert [row["username"] for row in contacts] == ["bob"]
    alice = await db_session.scalar(select(User).where(User.username == "alice"))
    bob = await db_session.scalar(select(User).where(User.username == "bob"))
    assert alice is not None and bob is not None
    stored = await db_session.scalar(
        select(Contact).where(Contact.owner_id == alice.id, Contact.contact_id == bob.id)
    )
    assert stored is not None


# Confirm adding the same username twice returns the existing edge.
async def test_add_contact_is_idempotent(client: AsyncClient) -> None:
    """Add Bob twice and require a single list entry."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    await _register_login(client, _BOB, key_fill=0x02)
    first = await client.post(
        "/contacts",
        json={"username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    second = await client.post(
        "/contacts",
        json={"username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    listed = await client.get("/contacts", headers={"Authorization": f"Bearer {alice_token}"})
    assert len(listed.json()["contacts"]) == 1


# Confirm an account cannot save itself as a contact.
async def test_cannot_add_self_as_contact(client: AsyncClient) -> None:
    """Ask Alice to add alice and require a 400."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    response = await client.post(
        "/contacts",
        json={"username": "alice"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 400


# Confirm an unknown username is not a user-existence oracle beyond 404.
async def test_add_unknown_contact_returns_404(client: AsyncClient) -> None:
    """Ask Alice to add a handle that was never registered."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    response = await client.post(
        "/contacts",
        json={"username": "nobody"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 404


# Confirm Alice's address book is not Bob's address book.
async def test_contacts_are_per_owner(client: AsyncClient) -> None:
    """Have Alice save Bob and require Bob's GET /contacts to stay empty."""

    alice_token = await _register_login(client, _ALICE, key_fill=0x01)
    bob_token = await _register_login(client, _BOB, key_fill=0x02)
    await client.post(
        "/contacts",
        json={"username": "bob"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    bob_list = await client.get("/contacts", headers={"Authorization": f"Bearer {bob_token}"})
    assert bob_list.status_code == 200
    assert bob_list.json()["contacts"] == []


# Confirm contact REST requires a bearer access token.
async def test_contacts_require_authentication(client: AsyncClient) -> None:
    """GET and POST /contacts with no Authorization header and require 401 or 403."""

    listed = await client.get("/contacts")
    added = await client.post("/contacts", json={"username": "bob"})
    assert listed.status_code in (401, 403)
    assert added.status_code in (401, 403)
