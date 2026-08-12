"""Expose ORM models so Alembic autogeneration can discover their metadata."""

# Import the User model so importing this package registers its table.
from app.models.user import User

# Declare the public re-export surface for this package.
__all__ = ["User"]
