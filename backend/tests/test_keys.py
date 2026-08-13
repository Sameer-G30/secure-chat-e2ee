"""Verify authenticated public-key upload and lookup, and rejection of bad input."""

# Import base64 to build well-formed and malformed test key payloads.
import base64

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

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


# Register and log an account in, returning its access token.
async def _register_and_get_access_token(client: AsyncClient, payload: dict[str, str]) -> str:
    """Create an account and log in, returning only its bearer access token."""

    await client.post("/auth/register", json=payload)
    response = await client.post(
        "/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    return str(response.json()["access_token"])


# Confirm an authenticated user can upload a well-formed public key.
async def test_upload_my_public_key_succeeds(client: AsyncClient) -> None:
    """Upload a valid key and require it echoed back under the caller's username."""

    access_token = await _register_and_get_access_token(client, _ALICE)
    response = await client.post(
        "/keys/me",
        json={"public_key": _fake_public_key()},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "alice"
    assert body["public_key"] == _fake_public_key()


# Confirm an unauthenticated upload attempt is rejected before touching the database.
async def test_upload_my_public_key_requires_authentication(client: AsyncClient) -> None:
    """Submit a key with no Authorization header and require a 401 or 403."""

    response = await client.post("/keys/me", json={"public_key": _fake_public_key()})
    # FastAPI's HTTPBearer(auto_error=True) reports a missing header as 403.
    assert response.status_code in (401, 403)


# Confirm a key of the wrong decoded length is rejected as a validation error.
async def test_upload_my_public_key_rejects_wrong_length(client: AsyncClient) -> None:
    """Submit a key that decodes to fewer than 32 bytes and require a 422."""

    access_token = await _register_and_get_access_token(client, _ALICE)
    too_short = base64.b64encode(b"too-short").decode("ascii")
    response = await client.post(
        "/keys/me",
        json={"public_key": too_short},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 422


# Confirm a non-base64 key payload is rejected as a validation error.
async def test_upload_my_public_key_rejects_invalid_base64(client: AsyncClient) -> None:
    """Submit clearly invalid base64 text and require a 422."""

    access_token = await _register_and_get_access_token(client, _ALICE)
    response = await client.post(
        "/keys/me",
        json={"public_key": "!!!not-base64!!!"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert response.status_code == 422


# Confirm one authenticated account can look up another account's uploaded public key.
async def test_get_public_key_returns_uploaded_key(client: AsyncClient) -> None:
    """Have Bob upload a key, then have Alice fetch it by username."""

    bob_access_token = await _register_and_get_access_token(client, _BOB)
    await client.post(
        "/keys/me",
        json={"public_key": _fake_public_key(0x02)},
        headers={"Authorization": f"Bearer {bob_access_token}"},
    )

    alice_access_token = await _register_and_get_access_token(client, _ALICE)
    response = await client.get(
        "/keys/bob", headers={"Authorization": f"Bearer {alice_access_token}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "bob"
    assert body["public_key"] == _fake_public_key(0x02)


# Confirm looking up a user who has not uploaded a key yet returns 404, not a stale value.
async def test_get_public_key_returns_404_before_upload(client: AsyncClient) -> None:
    """Register Bob but never upload his key, then require lookup to 404."""

    await client.post("/auth/register", json=_BOB)
    alice_access_token = await _register_and_get_access_token(client, _ALICE)
    response = await client.get(
        "/keys/bob", headers={"Authorization": f"Bearer {alice_access_token}"}
    )
    assert response.status_code == 404


# Confirm looking up a username that was never registered also returns 404.
async def test_get_public_key_returns_404_for_unknown_username(client: AsyncClient) -> None:
    """Look up a nonexistent username and require a 404, not a 500."""

    alice_access_token = await _register_and_get_access_token(client, _ALICE)
    response = await client.get(
        "/keys/nobody", headers={"Authorization": f"Bearer {alice_access_token}"}
    )
    assert response.status_code == 404


# Confirm an unauthenticated lookup attempt is rejected before touching the database.
async def test_get_public_key_requires_authentication(client: AsyncClient) -> None:
    """Look up a key with no Authorization header and require a 401 or 403."""

    response = await client.get("/keys/alice")
    assert response.status_code in (401, 403)
