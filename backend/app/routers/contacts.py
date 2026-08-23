"""Expose authenticated REST to save and list server-side contacts."""

# Import Annotated for dependency metadata.
from typing import Annotated

# Import FastAPI's routing and dependency primitives.
from fastapi import APIRouter, Depends

# Import SQLAlchemy's async session type used by the injected database dependency.
from sqlalchemy.ext.asyncio import AsyncSession

# Import the request-scoped database session dependency.
from app.db import get_db

# Import the ORM model the auth dependency returns.
from app.models.user import User

# Import the validated request and response shapes for this router's endpoints.
from app.schemas.contacts import AddContactRequest, ContactListResponse, ContactResponse

# Import the shared bearer-token authentication dependency.
from app.security.dependencies import get_current_user

# Import contact load/create helpers so routers stay thin.
from app.services.contacts import add_contact_for_owner, list_contacts_for_owner

# Group contact REST under one versionable tag; paths are absolute.
router = APIRouter(tags=["contacts"])


# Return the authenticated caller's server-side address book.
@router.get("/contacts", response_model=ContactListResponse)
async def list_contacts(
    # Require a valid access token; the address book is per-account, not localStorage.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ContactListResponse:
    """Return contacts the caller saved. Survives a new login on this account."""

    return await list_contacts_for_owner(db, current_user)


# Save a named account on the caller's address book.
@router.post("/contacts", response_model=ContactResponse)
async def add_contact(
    # Accept the contact's username; the owner is taken from the access token.
    payload: AddContactRequest,
    # Require a valid access token.
    current_user: Annotated[User, Depends(get_current_user)],
    # Inject a request-scoped async database session.
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ContactResponse:
    """Add a contact by username. Idempotent when the edge already exists."""

    return await add_contact_for_owner(db, current_user, payload.username)
