"""Verify login issues token pairs, rejects bad credentials, and is rate-limited."""

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import the token decoder used to assert on issued JWT claims directly.
from app.security.tokens import ACCESS_TOKEN_TYPE, REFRESH_TOKEN_TYPE, decode_token

# Reuse one valid registration payload shape across tests.
_VALID_PAYLOAD = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "correct horse battery staple",
}


# Register the shared test account once per test that needs to log in.
async def _register(client: AsyncClient) -> None:
    """Create the standard test account through the real registration endpoint."""

    response = await client.post("/auth/register", json=_VALID_PAYLOAD)
    assert response.status_code == 201


# Confirm a correct username/password pair returns a usable token pair.
async def test_login_returns_access_and_refresh_tokens(client: AsyncClient) -> None:
    """Log in after registering and assert both tokens verify with the right claims."""

    await _register(client)
    response = await client.post(
        "/auth/login",
        json={"username": _VALID_PAYLOAD["username"], "password": _VALID_PAYLOAD["password"]},
    )
    assert response.status_code == 200
    body = response.json()
    # Require both tokens and the bearer scheme name in the response.
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and body["access_token"]
    assert isinstance(body["refresh_token"], str) and body["refresh_token"]
    # A freshly registered account has not uploaded a key yet.
    assert body["has_public_key"] is False
    # Require the access token to verify and carry the right type/claims.
    access_claims = decode_token(body["access_token"], expected_type=ACCESS_TOKEN_TYPE)
    assert access_claims["username"] == "alice"
    # Require the refresh token to verify independently as a refresh-typed JWT.
    refresh_claims = decode_token(body["refresh_token"], expected_type=REFRESH_TOKEN_TYPE)
    assert refresh_claims["sub"] == access_claims["sub"]


# Confirm a wrong password is rejected without leaking which field was wrong.
async def test_login_rejects_wrong_password(client: AsyncClient) -> None:
    """Attempt login with an incorrect password and require a generic 401."""

    await _register(client)
    response = await client.post(
        "/auth/login", json={"username": _VALID_PAYLOAD["username"], "password": "wrong password"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid username or password"


# Confirm a nonexistent username produces the identical generic error as a wrong password.
async def test_login_rejects_unknown_username_with_same_generic_message(
    client: AsyncClient,
) -> None:
    """Attempt login against a username that was never registered."""

    response = await client.post(
        "/auth/login", json={"username": "nobody", "password": "does not matter at all"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid username or password"


# Confirm the login endpoint enforces the spec's rate-limiting requirement.
async def test_login_is_rate_limited(client: AsyncClient) -> None:
    """Exceed the configured per-minute login limit and require an eventual 429."""

    await _register(client)
    statuses = []
    for _ in range(11):
        response = await client.post(
            "/auth/login",
            json={"username": _VALID_PAYLOAD["username"], "password": "wrong password"},
        )
        statuses.append(response.status_code)
    # Require that the eleventh attempt from the same client address was throttled.
    assert statuses[-1] == 429
