"""Provide the async SQLAlchemy engine, session factory, and ORM base."""

# Import AsyncIterator to type the FastAPI dependency generator correctly.
from collections.abc import AsyncIterator

# Import the async engine factory, session factory, and session/engine types.
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Import the declarative base class every ORM model must inherit from.
from sqlalchemy.orm import DeclarativeBase

# Import validated settings so the engine URL always comes from the environment.
from app.config import get_settings


# Declare the shared metadata root used by Alembic autogeneration and models.
class Base(DeclarativeBase):
    """Serve as the declarative base for every ORM-mapped table."""


# Build the module-wide async engine once, from validated settings.
def _build_engine() -> AsyncEngine:
    """Create the process-wide async engine from the configured database URL."""

    # Read the same validated URL used by Alembic and the health endpoint.
    settings = get_settings()
    # Construct the async engine without eagerly opening a connection.
    return create_async_engine(settings.database_url, pool_pre_ping=True)


# Hold one engine instance for the lifetime of the process.
engine: AsyncEngine = _build_engine()

# Build a session factory bound to the shared engine.
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


# Provide a request-scoped database session to path operations.
async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield an async session and guarantee it closes after the request."""

    # Open one session per request rather than sharing sessions across requests.
    async with async_session_maker() as session:
        # Hand the session to the endpoint; FastAPI resumes here after the response.
        yield session
