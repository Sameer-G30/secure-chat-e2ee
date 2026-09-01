"""Expose ORM models so Alembic autogeneration can discover their metadata."""

# Import the Block model so importing this package registers its table.
from app.models.block import Block

# Import the Contact model so importing this package registers its table.
from app.models.contact import Contact

# Import the Conversation model so importing this package registers its table.
from app.models.conversation import Conversation

# Import the Message model so importing this package registers its table.
from app.models.message import Message

# Import the MessageHide model so importing this package registers its table.
from app.models.message_hide import MessageHide

# Import the RefreshToken model so importing this package registers its table.
from app.models.refresh_token import RefreshToken

# Import the Report model so importing this package registers its table.
from app.models.report import Report

# Import the User model so importing this package registers its table.
from app.models.user import User

# Declare the public re-export surface for this package.
__all__ = [
    "Block",
    "Contact",
    "Conversation",
    "Message",
    "MessageHide",
    "RefreshToken",
    "Report",
    "User",
]
