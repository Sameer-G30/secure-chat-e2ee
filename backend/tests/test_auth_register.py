"""Verify Argon2id registration, uniqueness enforcement, and rate limiting."""

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import the ORM session/model to assert on stored data directly.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.security.passwords import verify_password

# Reuse one valid payload shape across tests, varying only what each test needs.
_VALID_PAYLOAD = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "correct horse battery staple",
}


# Confirm a valid registration succeeds and stores an Argon2id hash, not plaintext.
async def test_register_creates_account_with_argon2id_hash(client: AsyncClient) -> None:
    """Register once and assert the response and stored hash are both correct."""

    # Submit a valid registration payload.
    response = await client.post("/auth/register", json=_VALID_PAYLOAD)
    # Require the created-resource status code.
    assert response.status_code == 201
    # Parse the JSON body returned to the client.
    body = response.json()
    # Confirm the response echoes the submitted identity fields.
    assert body["username"] == "alice"
    assert body["email"] == "alice@example.com"
    # Confirm no password or hash field ever reaches the client response.
    assert "password" not in body
    assert "password_hash" not in body
    # Confirm no public key is present yet; that lands with Slice 3's key upload.
    assert "public_key" not in body


# Confirm the stored hash is Argon2id and never the submitted plaintext.
async def test_register_never_stores_plaintext_password(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Load the persisted row and assert it holds a verifiable Argon2id hash."""

    # Register the account through the public API.
    await client.post("/auth/register", json=_VALID_PAYLOAD)
    # Load the row directly through the same test database session.
    user = await db_session.scalar(select(User).where(User.username == "alice"))
    # Require that registration actually persisted a row.
    assert user is not None
    # Require the stored value to be an Argon2id hash, not the plaintext password.
    assert user.password_hash != _VALID_PAYLOAD["password"]
    assert user.password_hash.startswith("$argon2id$")
    # Require the hash to verify against the original password.
    assert verify_password(_VALID_PAYLOAD["password"], user.password_hash)
    # Require the public key column to start null, ahead of Slice 3's upload endpoint.
    assert user.public_key is None


# Confirm a duplicate username is rejected with a specific conflict response.
async def test_register_rejects_duplicate_username(client: AsyncClient) -> None:
    """Register once, then attempt a second account with the same username."""

    # Register the first account successfully.
    await client.post("/auth/register", json=_VALID_PAYLOAD)
    # Attempt a second registration reusing the username with a different email.
    duplicate = {**_VALID_PAYLOAD, "email": "different@example.com"}
    response = await client.post("/auth/register", json=duplicate)
    # Require a conflict rather than a duplicate row or a server error.
    assert response.status_code == 409
    assert "username" in response.json()["detail"]


# Confirm a duplicate email is rejected with a specific conflict response.
async def test_register_rejects_duplicate_email(client: AsyncClient) -> None:
    """Register once, then attempt a second account with the same email."""

    # Register the first account successfully.
    await client.post("/auth/register", json=_VALID_PAYLOAD)
    # Attempt a second registration reusing the email with a different username.
    duplicate = {**_VALID_PAYLOAD, "username": "alice2"}
    response = await client.post("/auth/register", json=duplicate)
    # Require a conflict rather than a duplicate row or a server error.
    assert response.status_code == 409
    assert "email" in response.json()["detail"]


# Confirm passwords shorter than the documented minimum are rejected before hashing.
async def test_register_rejects_short_password(client: AsyncClient) -> None:
    """Submit a too-short password and require a validation error, not a 500."""

    # Build a payload with a password below the eight-character floor.
    weak_payload = {**_VALID_PAYLOAD, "password": "short"}
    response = await client.post("/auth/register", json=weak_payload)
    # FastAPI/Pydantic reports request validation failures as 422.
    assert response.status_code == 422


# Confirm malformed email addresses are rejected before hashing or a database write.
async def test_register_rejects_invalid_email(client: AsyncClient) -> None:
    """Submit a syntactically invalid email and require a validation error."""

    # Build a payload with an email missing the required "@" structure.
    invalid_payload = {**_VALID_PAYLOAD, "email": "not-an-email"}
    response = await client.post("/auth/register", json=invalid_payload)
    # FastAPI/Pydantic reports request validation failures as 422.
    assert response.status_code == 422


# Confirm the registration endpoint enforces the spec's rate-limiting requirement.
async def test_register_is_rate_limited(client: AsyncClient) -> None:
    """Exceed the configured per-minute limit and require an eventual 429."""

    # Send one more request than the configured "5/minute" register limit allows.
    statuses = []
    for attempt in range(6):
        # Vary identity fields so the 429 is caused by the rate limit, not a conflict.
        payload = {
            "username": f"limituser{attempt}",
            "email": f"limituser{attempt}@example.com",
            "password": "correct horse battery staple",
        }
        response = await client.post("/auth/register", json=payload)
        statuses.append(response.status_code)
    # Require that the sixth attempt from the same client address was throttled.
    assert statuses[-1] == 429
