"""Verify the first operational API contract."""

# Import asyncio to run the asynchronous in-process HTTP request.
import asyncio

# Import HTTPX's ASGI transport and asynchronous client.
from httpx import ASGITransport, AsyncClient

# Import the application under test.
from app.main import app


# Exercise the ASGI application without opening a network socket.
async def request_health() -> tuple[int, dict[str, str]]:
    """Request the health contract through HTTPX's in-process ASGI transport."""

    # Route client requests directly through the FastAPI ASGI application.
    transport = ASGITransport(app=app)
    # Manage the asynchronous client's connection lifecycle explicitly.
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Send the same GET request used by container health probes.
        response = await client.get("/health")
    # Return stable primitives so the synchronous test remains straightforward.
    return response.status_code, response.json()


# Confirm that deployment health checks receive a stable success response.
def test_health_endpoint_reports_ready() -> None:
    """Return a versioned healthy response without exposing secrets."""

    # Run the in-process asynchronous request to completion.
    status_code, payload = asyncio.run(request_health())
    # Require the standard HTTP success status.
    assert status_code == 200
    # Confirm the process reports itself as available.
    assert payload["status"] == "ok"
    # Confirm the response identifies this first API slice.
    assert payload["version"] == "0.1.0"
    # Confirm no database URL or future secret appears in diagnostics.
    assert "database_url" not in payload
