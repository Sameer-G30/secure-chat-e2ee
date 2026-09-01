"""Shared timezone normalization used anywhere a naive DB timestamp meets an aware clock.

Extracted during the pre-deployment refactor: `app/routers/auth.py` and
`app/services/epoch.py` each defined their own byte-for-byte identical
`as_utc()` helper for the same reason (SQLite silently drops tzinfo on a
`DateTime(timezone=True)` column even though Postgres keeps it). One shared
implementation means the "every value this project writes is already UTC"
assumption only has to be documented, and kept correct, in one place.
"""

# Import the timezone-aware datetime primitives this helper normalizes between.
from datetime import UTC, datetime


def as_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime, treating a naive value as already-UTC.

    SQLite often returns naive datetimes even when the column is declared
    `DateTime(timezone=True)`; Postgres does not. Every timestamp this project
    writes is generated as UTC (`datetime.now(UTC)` or the database's
    `now()`), so a naive value read back is safely reinterpreted as UTC
    rather than guessing a local zone.
    """

    if value.tzinfo is None:
        # Attach UTC to a naive datetime instead of guessing a local zone.
        return value.replace(tzinfo=UTC)
    # Convert any other offset-aware value into UTC before comparison/subtraction.
    return value.astimezone(UTC)
