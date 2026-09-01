"""Verify metadata-only abuse reports. Never accepts or stores message content."""

# Import UUID to compare a JSON response id against the stored Uuid column.
from uuid import UUID

# Import HTTPX's async client type for fixture-provided request calls.
from httpx import AsyncClient

# Import SQLAlchemy helpers to inspect stored report rows.
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import ORM models used for direct row assertions.
from app.models.report import Report
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


async def _register_login(client: AsyncClient, payload: dict[str, str]) -> str:
    """Create an account and return its access token."""

    await client.post("/auth/register", json=payload)
    response = await client.post(
        "/auth/login", json={"username": payload["username"], "password": payload["password"]}
    )
    return str(response.json()["access_token"])


# Confirm a report is persisted with reporter/reported metadata and a reason.
async def test_file_report_persists_metadata_only(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """File a report against Bob and require a metadata-only stored row."""

    alice_token = await _register_login(client, _ALICE)
    await _register_login(client, _BOB)

    response = await client.post(
        "/reports",
        json={"username": "bob", "reason": "sent a phishing link"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 201
    body = response.json()
    assert "id" in body
    assert set(body.keys()) == {"id", "created_at"}

    alice = await db_session.scalar(select(User).where(User.username == "alice"))
    bob = await db_session.scalar(select(User).where(User.username == "bob"))
    assert alice is not None and bob is not None
    stored = await db_session.scalar(select(Report).where(Report.id == UUID(body["id"])))
    assert stored is not None
    assert stored.reporter_id == alice.id
    assert stored.reported_id == bob.id
    assert stored.reason == "sent a phishing link"


# Confirm repeated reports about ongoing abuse are not collapsed into one row.
async def test_filing_the_same_report_twice_creates_two_rows(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """File the same report twice and require two distinct rows, unlike contacts/blocks."""

    alice_token = await _register_login(client, _ALICE)
    await _register_login(client, _BOB)
    headers = {"Authorization": f"Bearer {alice_token}"}

    first = await client.post(
        "/reports", json={"username": "bob", "reason": "spam"}, headers=headers
    )
    second = await client.post(
        "/reports", json={"username": "bob", "reason": "spam"}, headers=headers
    )
    assert first.json()["id"] != second.json()["id"]
    stored = list((await db_session.scalars(select(Report))).all())
    assert len(stored) == 2


# Confirm an account cannot report itself.
async def test_cannot_report_self(client: AsyncClient) -> None:
    """Ask Alice to report alice and require a 400."""

    alice_token = await _register_login(client, _ALICE)
    response = await client.post(
        "/reports",
        json={"username": "alice", "reason": "testing"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 400


# Confirm reporting an unknown username is not a user-existence oracle beyond 404.
async def test_report_unknown_user_returns_404(client: AsyncClient) -> None:
    """Report a handle that was never registered."""

    alice_token = await _register_login(client, _ALICE)
    response = await client.post(
        "/reports",
        json={"username": "nobody", "reason": "testing"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 404


# Confirm an empty reason is rejected before it ever reaches storage.
async def test_report_requires_a_nonempty_reason(client: AsyncClient) -> None:
    """File a report with an empty reason string and require a validation error."""

    alice_token = await _register_login(client, _ALICE)
    await _register_login(client, _BOB)
    response = await client.post(
        "/reports",
        json={"username": "bob", "reason": ""},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 422


# Confirm an oversized reason is rejected rather than silently truncated.
async def test_report_reason_is_length_bounded(client: AsyncClient) -> None:
    """File a report with a 501-character reason and require a validation error."""

    alice_token = await _register_login(client, _ALICE)
    await _register_login(client, _BOB)
    response = await client.post(
        "/reports",
        json={"username": "bob", "reason": "x" * 501},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 422


# Confirm no field exists for attaching message content to a report.
async def test_report_payload_has_no_message_content_field(client: AsyncClient) -> None:
    """Send an extra message-content-shaped field and require it to be ignored, not stored."""

    alice_token = await _register_login(client, _ALICE)
    await _register_login(client, _BOB)
    response = await client.post(
        "/reports",
        json={
            "username": "bob",
            "reason": "phishing",
            "message_ciphertext": "should be ignored",
            "plaintext": "should never be accepted",
        },
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert response.status_code == 201
    assert set(response.json().keys()) == {"id", "created_at"}


# Confirm the report endpoint requires a bearer access token.
async def test_report_requires_authentication(client: AsyncClient) -> None:
    """File a report with no Authorization header and require 401 or 403."""

    response = await client.post("/reports", json={"username": "bob", "reason": "testing"})
    assert response.status_code in (401, 403)
