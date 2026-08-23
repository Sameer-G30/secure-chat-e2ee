"""Validate contact create/list payloads without carrying message bodies or scores."""

# Import datetime for the response's typed timestamp field.
from datetime import datetime

# Import UUID for typed account identifiers.
from uuid import UUID

# Import Pydantic's model base and field-level validation helpers.
from pydantic import BaseModel, Field

# Reuse the same username bounds the conversation schema already documents.
_USERNAME_MIN_LENGTH = 3
_USERNAME_MAX_LENGTH = 32


# Validate the single field an add-contact request must supply.
class AddContactRequest(BaseModel):
    """Represent the client-submitted handle to save on the owner's address book."""

    # Identify the other account by the same handle used for GET /keys/{username}.
    username: str = Field(min_length=_USERNAME_MIN_LENGTH, max_length=_USERNAME_MAX_LENGTH)


# Describe one saved contact as returned to the authenticated owner.
class ContactResponse(BaseModel):
    """Represent one address-book entry without any private key or message body."""

    # Identify the contact account with the UUID used as AEAD associated data.
    id: UUID
    # Identify the contact account with the handle the sidebar displays.
    username: str
    # Record when the owner saved this contact.
    created_at: datetime


# Describe the owner's full server-side address book.
class ContactListResponse(BaseModel):
    """Represent every contact row owned by the authenticated caller."""

    # Carry the saved contacts; empty when the owner has not added anyone yet.
    contacts: list[ContactResponse]
