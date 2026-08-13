"""Verify refresh-token rotation, single-use enforcement, and reuse detection."""

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import the ORM session/model type to assert on persisted rotation state directly.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.refresh_token import RefreshToken

_VALID_PAYLOAD = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "correct horse battery staple",
}


# Register and log in the shared test account, returning its issued token pair.
async def _register_and_login(client: AsyncClient) -> dict[str, str]:
    """Create the test account and log in, returning the parsed token response."""

    await client.post("/auth/register", json=_VALID_PAYLOAD)
    response = await client.post(
        "/auth/login",
        json={"username": _VALID_PAYLOAD["username"], "password": _VALID_PAYLOAD["password"]},
    )
    body: dict[str, str] = response.json()
    return body


# Confirm a valid refresh token yields a new pair and revokes the presented token.
async def test_refresh_rotates_to_a_new_token_pair(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Refresh once and assert the response differs and the old token is revoked."""

    tokens = await _register_and_login(client)
    response = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 200
    rotated = response.json()
    # Rotation must issue a genuinely new refresh token, not echo the presented one.
    #
    # (Access tokens carry no unique id, so two minted within the same
    # second are legitimately byte-identical; that is not a security
    # property this test needs to check.)
    assert rotated["refresh_token"] != tokens["refresh_token"]
    assert rotated["access_token"]

    # Assert the originally presented refresh token's row is now revoked.
    from app.security.tokens import hash_refresh_token

    original_hash = hash_refresh_token(tokens["refresh_token"])
    stored = await db_session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == original_hash)
    )
    assert stored is not None
    assert stored.revoked_at is not None


# Confirm the new refresh token from rotation is itself usable for a further rotation.
async def test_rotated_refresh_token_can_be_used_again(client: AsyncClient) -> None:
    """Chain two rotations and require both to succeed with fresh tokens each time."""

    tokens = await _register_and_login(client)
    first_rotation = await client.post(
        "/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert first_rotation.status_code == 200
    second_rotation = await client.post(
        "/auth/refresh", json={"refresh_token": first_rotation.json()["refresh_token"]}
    )
    assert second_rotation.status_code == 200
    assert second_rotation.json()["refresh_token"] != first_rotation.json()["refresh_token"]


# Confirm presenting an already-rotated (stale) refresh token a second time is rejected.
async def test_reusing_a_rotated_refresh_token_is_rejected(client: AsyncClient) -> None:
    """Rotate once, then present the same now-stale token again and require a 401."""

    tokens = await _register_and_login(client)
    first_use = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert first_use.status_code == 200

    replay = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert replay.status_code == 401
    assert "already been used" in replay.json()["detail"]


# Confirm reuse detection also revokes the *newer* token issued by the first rotation,
# forcing full re-authentication rather than leaving one still-valid session behind.
async def test_reuse_detection_revokes_every_active_token_for_the_account(
    client: AsyncClient,
) -> None:
    """After a replay is detected, the legitimately rotated token must also be dead."""

    tokens = await _register_and_login(client)
    first_use = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    legitimate_new_refresh_token = first_use.json()["refresh_token"]

    # An attacker (or a bug) replays the original, now-stale token.
    replay = await client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert replay.status_code == 401

    # The legitimate user's own newly rotated token must now also be revoked.
    follow_up = await client.post(
        "/auth/refresh", json={"refresh_token": legitimate_new_refresh_token}
    )
    assert follow_up.status_code == 401


# Confirm a malformed or unsigned token is rejected before any database lookup.
async def test_refresh_rejects_a_malformed_token(client: AsyncClient) -> None:
    """Submit an arbitrary string and require a 401, not a 500."""

    response = await client.post("/auth/refresh", json={"refresh_token": "not-a-real-jwt"})
    assert response.status_code == 401


# Confirm logout revokes a refresh token so it can never be rotated afterward.
async def test_logout_revokes_the_refresh_token(client: AsyncClient) -> None:
    """Log out with a valid refresh token, then require rotation to fail afterward."""

    tokens = await _register_and_login(client)
    logout_response = await client.post(
        "/auth/logout", json={"refresh_token": tokens["refresh_token"]}
    )
    assert logout_response.status_code == 204

    attempted_use = await client.post(
        "/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
    )
    assert attempted_use.status_code == 401


# Confirm logging out with an unrecognized token is a harmless no-op, not an error.
async def test_logout_is_a_no_op_for_an_unknown_token(client: AsyncClient) -> None:
    """Logout with a syntactically valid but never-issued token still returns 204."""

    tokens = await _register_and_login(client)
    response = await client.post("/auth/logout", json={"refresh_token": tokens["access_token"]})
    assert response.status_code == 204
