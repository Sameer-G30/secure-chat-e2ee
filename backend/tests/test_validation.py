"""Slice 9 inbound-validation sweep: empty handles, oversized fields, unsafe usernames."""

# Import base64 to build well-formed public-key payloads for authenticated tests.
import base64

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import the refresh-token length cap so the oversized-body test stays in sync.
from app.schemas.auth import REFRESH_TOKEN_MAX_LENGTH

# Reuse one valid registration payload, varying only the field under test.
_VALID_REGISTER = {
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

    # Repeat a fill byte to 32 raw bytes, then encode as standard base64.
    return base64.b64encode(bytes([fill_byte]) * 32).decode("ascii")


# Register, log in, and upload a public key, returning the access token.
async def _register_login(client: AsyncClient, payload: dict[str, str]) -> str:
    """Create an account, log in, and complete key upload."""

    # Create the account through the public registration contract.
    await client.post("/auth/register", json=payload)
    # Exchange credentials for an access token.
    response = await client.post(
        "/auth/login",
        json={"username": payload["username"], "password": payload["password"]},
    )
    # Read the bearer token the later authenticated calls need.
    access_token = str(response.json()["access_token"])
    # Upload a public key so conversation create is not rejected for a missing key.
    await client.post(
        "/keys/me",
        json={"public_key": _fake_public_key()},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    # Return the token for Authorization headers.
    return access_token


# Confirm an empty username is rejected at the schema layer, not hashed.
async def test_register_rejects_empty_username(client: AsyncClient) -> None:
    """Submit a blank handle and require 422 rather than a 500 or a stored row."""

    # Copy the valid payload and blank the username.
    payload = {**_VALID_REGISTER, "username": ""}
    # POST the invalid body to the public registration endpoint.
    response = await client.post("/auth/register", json=payload)
    # Pydantic/FastAPI reports request validation failures as 422.
    assert response.status_code == 422


# Confirm whitespace-only usernames fail the URL-safe character rule.
async def test_register_rejects_whitespace_username(client: AsyncClient) -> None:
    """Submit three spaces (length 3) and require 422 because spaces are not URL-safe."""

    # Three spaces would pass a naive min_length=3 check without the character rule.
    payload = {**_VALID_REGISTER, "username": "   "}
    # POST the invalid body to registration.
    response = await client.post("/auth/register", json=payload)
    # Require a validation error, not a stored account named three spaces.
    assert response.status_code == 422


# Confirm punctuation-heavy handles cannot be used as /keys/{username} path segments.
async def test_register_rejects_unsafe_username_characters(client: AsyncClient) -> None:
    """Submit a handle with '!' and require 422."""

    # Build a payload whose username would need percent-encoding in a URL path.
    payload = {**_VALID_REGISTER, "username": "alice!"}
    # POST the invalid body to registration.
    response = await client.post("/auth/register", json=payload)
    # Require a validation error before uniqueness checks run.
    assert response.status_code == 422


# Confirm an oversized email is rejected before hashing or a database write.
async def test_register_rejects_oversized_email(client: AsyncClient) -> None:
    """Submit a mailbox string longer than 254 characters and require 422."""

    # Build a syntactically dotted mailbox that exceeds the RFC 5321 length cap.
    oversized_email = ("a" * 243) + "@example.com"
    # Copy the valid payload and replace the email.
    payload = {**_VALID_REGISTER, "username": "alicesize", "email": oversized_email}
    # POST the oversized body to registration.
    response = await client.post("/auth/register", json=payload)
    # Require a validation error rather than storing an unbounded mailbox.
    assert response.status_code == 422


# Confirm login does not accept a missing username field value.
async def test_login_rejects_empty_username(client: AsyncClient) -> None:
    """Submit an empty login handle and require 422, not a generic 401."""

    # Empty string is a malformed body, not "wrong password".
    response = await client.post(
        "/auth/login",
        json={"username": "", "password": "correct horse battery staple"},
    )
    # Schema validation must run before Argon2id verification.
    assert response.status_code == 422


# Confirm refresh/logout cannot be used to POST an unbounded token string.
async def test_refresh_rejects_oversized_token(client: AsyncClient) -> None:
    """Submit a refresh_token longer than the documented cap and require 422."""

    # Build a body that is one byte over the cap; it need not be a real JWT.
    oversized = "a" * (REFRESH_TOKEN_MAX_LENGTH + 1)
    # POST the oversized token to the rotation endpoint.
    response = await client.post("/auth/refresh", json={"refresh_token": oversized})
    # Require rejection before JWT decode or a database lookup.
    assert response.status_code == 422


# Confirm add-contact uses the same URL-safe username rule as register.
async def test_add_contact_rejects_empty_and_unsafe_usernames(client: AsyncClient) -> None:
    """POST /contacts with blank, whitespace, and unsafe handles; all must 422."""

    # Authenticate Alice so the 422 is from the body, not missing auth.
    alice_token = await _register_login(client, _VALID_REGISTER)
    # Try an empty handle first.
    empty = await client.post(
        "/contacts",
        json={"username": ""},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    # Require schema rejection.
    assert empty.status_code == 422
    # Try a whitespace handle that would pass min_length=3 without the character rule.
    whitespace = await client.post(
        "/contacts",
        json={"username": "   "},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    # Require schema rejection rather than a 404 user-not-found oracle on spaces.
    assert whitespace.status_code == 422
    # Try a punctuation handle that cannot be a registered username.
    unsafe = await client.post(
        "/contacts",
        json={"username": "bob!"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    # Require schema rejection.
    assert unsafe.status_code == 422


# Confirm start-conversation rejects empty and unsafe peer handles.
async def test_create_conversation_rejects_empty_and_unsafe_peer(client: AsyncClient) -> None:
    """POST /conversations with blank and unsafe peer_username values; both must 422."""

    # Authenticate Alice so conversation create is otherwise allowed.
    alice_token = await _register_login(client, _VALID_REGISTER)
    # Also create Bob so a valid handle would succeed; we only send invalid ones.
    await _register_login(client, _BOB)
    # Try an empty peer handle.
    empty = await client.post(
        "/conversations",
        json={"peer_username": ""},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    # Require schema rejection.
    assert empty.status_code == 422
    # Try an unsafe peer handle.
    unsafe = await client.post(
        "/conversations",
        json={"peer_username": "bob!"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    # Require schema rejection rather than a 404 that looks like "no such user".
    assert unsafe.status_code == 422


# Confirm public-key lookup rejects oversized and unsafe path usernames.
async def test_get_public_key_rejects_oversized_and_unsafe_username(
    client: AsyncClient,
) -> None:
    """GET /keys/{username} with a 33-char handle and a '!' handle; both must 422."""

    # Authenticate so the 422 is from Path validation, not missing auth.
    alice_token = await _register_login(client, _VALID_REGISTER)
    # Build a handle one character over the documented cap.
    oversized = "a" * 33
    # Look up the oversized path segment.
    long_name = await client.get(
        f"/keys/{oversized}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    # Require Path validation to fail closed.
    assert long_name.status_code == 422
    # Look up a handle that cannot be registered.
    unsafe = await client.get(
        "/keys/alice!",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    # Require Path pattern rejection rather than a 404.
    assert unsafe.status_code == 422
