"""Provide a fast, isolated database and HTTP client for backend tests."""

# Import AsyncIterator to type async fixture generators.
from collections.abc import AsyncIterator

# Import pytest and pytest-asyncio's fixture decorator.
import pytest

# Import HTTPX's ASGI transport and asynchronous client for in-process requests.
from httpx import ASGITransport, AsyncClient

# Import the async engine/session types and factories used only by tests.
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# Import every ORM model so their tables register on Base.metadata before create_all.
from app import models  # noqa: F401
from app.db import Base, get_db
from app.main import app
from app.security.rate_limit import limiter

# Use an in-memory SQLite database so tests never require a running Postgres
# container or network access; StaticPool keeps the single in-memory database
# alive across the multiple connections FastAPI's async session opens.
_TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# Build one private engine and schema per test function.
@pytest.fixture
async def _engine() -> AsyncIterator[AsyncEngine]:
    """Yield a fresh in-memory SQLite engine with all tables created."""

    # Build a private engine so tests never touch the real development database.
    test_engine = create_async_engine(_TEST_DATABASE_URL, poolclass=StaticPool)
    # Create every table fresh for this test.
    async with test_engine.begin() as connection:
        # Run the synchronous DDL API through the async connection bridge.
        await connection.run_sync(Base.metadata.create_all)
    # Hand the ready engine to dependent fixtures.
    yield test_engine
    # Dispose the private engine's connections once the test finishes.
    await test_engine.dispose()


# Expose a session bound to the test engine for assertions that bypass the API.
@pytest.fixture
async def db_session(_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Yield a session tests can use to inspect rows the API created."""

    # Build a session factory bound to the private test engine.
    session_maker = async_sessionmaker(_engine, expire_on_commit=False)
    # Open one session for the duration of the test.
    async with session_maker() as session:
        # Hand the session to the test function.
        yield session


# Provide one FastAPI test client per test, backed by a fresh in-memory schema.
@pytest.fixture
async def client(_engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    """Yield an HTTP client wired to an isolated in-memory database."""

    # Build a session factory bound to the same private test engine.
    session_maker = async_sessionmaker(_engine, expire_on_commit=False)

    # Override the app's database dependency with the isolated test session.
    async def _get_test_db() -> AsyncIterator[AsyncSession]:
        """Yield a session bound to the private in-memory test engine."""

        # Open one session per request, mirroring the production dependency.
        async with session_maker() as session:
            # Hand the session to the endpoint under test.
            yield session

    # Swap in the test database dependency for the duration of this test.
    app.dependency_overrides[get_db] = _get_test_db
    # Reset slowapi's in-memory counters so earlier tests cannot exhaust a later one's limit.
    limiter.reset()

    # Route requests through the ASGI app without opening a real network socket.
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        # Hand control to the test function.
        yield test_client

    # Remove the override so later tests are unaffected by this fixture.
    app.dependency_overrides.pop(get_db, None)
