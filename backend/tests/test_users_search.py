"""Verify authenticated username search replaces the legacy full-table-scan approach."""

# Import base64 to build well-formed public-key payloads for test accounts.
import base64

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

_ALICE = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "correct horse battery staple",
}
_ALICIA = {
    "username": "alicia",
    "email": "alicia@example.com",
    "password": "another strong passphrase!!",
}
_BOB = {
    "username": "bob",
    "email": "bob@example.com",
    "password": "bob has a strong passphrase",
}


# Build a syntactically valid base64 X25519-sized (32-byte) public key for test accounts.
def _fake_public_key(fill_byte: int = 0x01) -> str:
    """Return base64 text decoding to exactly 32 bytes, as a real key would."""

    return base64.b64encode(bytes([fill_byte]) * 32).decode("ascii")


# Register and log in, returning the access token.
async def _register_login(client: AsyncClient, payload: dict[str, str]) -> str:
    """Create an account and return its access token; key upload is not required to search."""

    await client.post("/auth/register", json=payload)
    response = await client.post(
        "/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    return str(response.json()["access_token"])


# Confirm a case-insensitive prefix match returns every matching account.
async def test_search_matches_case_insensitive_prefix(client: AsyncClient) -> None:
    """Search "ALI" and require both alice and alicia, never bob."""

    token = await _register_login(client, _BOB)
    await _register_login(client, _ALICE)
    await _register_login(client, _ALICIA)

    response = await client.get(
        "/users/search", params={"q": "ALI"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    usernames = {row["username"] for row in response.json()["users"]}
    assert usernames == {"alice", "alicia"}


# Confirm the response never leaks email, password hash, or public key.
async def test_search_result_shape_is_username_only(client: AsyncClient) -> None:
    """Require exactly {"username": ...} per row, matching app/schemas/users.py."""

    token = await _register_login(client, _BOB)
    await _register_login(client, _ALICE)

    response = await client.get(
        "/users/search", params={"q": "ali"}, headers={"Authorization": f"Bearer {token}"}
    )
    row = response.json()["users"][0]
    assert set(row.keys()) == {"username"}


# Confirm the caller never finds their own account in search results.
async def test_search_excludes_the_caller(client: AsyncClient) -> None:
    """Have Alice search for her own prefix and require an empty result."""

    token = await _register_login(client, _ALICE)
    response = await client.get(
        "/users/search", params={"q": "ali"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.json()["users"] == []


# Confirm a query shorter than the documented minimum is rejected, not silently truncated.
async def test_search_requires_minimum_query_length(client: AsyncClient) -> None:
    """Search with a single character and require a validation error."""

    token = await _register_login(client, _ALICE)
    response = await client.get(
        "/users/search", params={"q": "a"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 422


# Confirm the result count never exceeds the documented server-side cap, even if asked.
async def test_search_limit_is_bounded(client: AsyncClient) -> None:
    """Ask for a limit above MAX_USER_SEARCH_RESULTS and require a validation error."""

    token = await _register_login(client, _ALICE)
    response = await client.get(
        "/users/search",
        params={"q": "al", "limit": 1000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


# Confirm a query matching nobody returns an empty list, not an error.
async def test_search_with_no_matches_returns_empty_list(client: AsyncClient) -> None:
    """Search for a prefix nobody has and require an empty, successful response."""

    token = await _register_login(client, _ALICE)
    response = await client.get(
        "/users/search", params={"q": "zzz"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["users"] == []


# Confirm the endpoint requires a bearer access token.
async def test_search_requires_authentication(client: AsyncClient) -> None:
    """Search with no Authorization header and require 401 or 403."""

    response = await client.get("/users/search", params={"q": "al"})
    assert response.status_code in (401, 403)
