"""Load and persist server-side contacts without storing message bodies or scores."""

# Import FastAPI's HTTP-error primitives so callers can return stable status codes.
from fastapi import HTTPException, status

# Import SQLAlchemy query helpers and the uniqueness-conflict error.
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# Import the ORM models this service reads and writes.
from app.models.contact import Contact
from app.models.user import User

# Import the response shapes so routers do not rebuild contact payloads.
from app.schemas.contacts import ContactListResponse, ContactResponse

# Import unread counts so the sidebar badge is server-synced metadata, not a preview.
from app.services.receipts import unread_counts_for_owner

# Shared detail when the named account does not exist.
_USER_NOT_FOUND_DETAIL = "user not found"
# Shared detail when the caller tries to save their own account.
_SELF_CONTACT_DETAIL = "cannot add yourself as a contact"


# Build the client-facing contact payload from ORM rows.
def serialize_contact(
    contact: Contact,
    peer: User,
    *,
    unread_count: int = 0,
) -> ContactResponse:
    """Return the contact's user id, handle, profile flags, and unread badge."""

    return ContactResponse(
        id=peer.id,
        username=peer.username,
        display_name=peer.display_name,
        has_avatar=peer.avatar_bytes is not None,
        unread_count=unread_count,
        created_at=contact.created_at,
    )


# Return every contact the authenticated owner has saved, newest first.
async def list_contacts_for_owner(db: AsyncSession, owner: User) -> ContactListResponse:
    """Load the owner's address book joined to usernames; never a localStorage list."""

    # Join contacts to users so the sidebar can render handles without a second round trip.
    rows = (
        await db.execute(
            select(Contact, User)
            .join(User, User.id == Contact.contact_id)
            .where(Contact.owner_id == owner.id)
            .order_by(Contact.created_at.desc())
        )
    ).all()
    # Serialize each joined pair into the public response shape.
    peer_ids = [peer.id for _contact, peer in rows]
    unread = await unread_counts_for_owner(db, owner, peer_ids)
    items = [
        serialize_contact(contact, peer, unread_count=unread.get(peer.id, 0))
        for contact, peer in rows
    ]
    return ContactListResponse(contacts=items)


# Save a named account on the owner's address book, idempotently.
async def add_contact_for_owner(db: AsyncSession, owner: User, username: str) -> ContactResponse:
    """Insert (owner_id, contact_id) or return the existing row.

    Unknown usernames 404. Self-add 400. A duplicate add is 200 with the
    existing row so the UI can retry safely.
    """

    # An address-book edge to the caller's own account is not a 1:1 contact.
    if username == owner.username:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=_SELF_CONTACT_DETAIL)

    # Look up the named account by the same unique handle conversations use.
    peer = await db.scalar(select(User).where(User.username == username))
    if peer is None:
        # Do not distinguish "never registered" from other lookup misses.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_USER_NOT_FOUND_DETAIL)

    # Reuse an existing edge when the owner already saved this account.
    existing = await db.scalar(
        select(Contact).where(Contact.owner_id == owner.id, Contact.contact_id == peer.id)
    )
    if existing is not None:
        return serialize_contact(existing, peer)

    # Insert a new edge; a concurrent request may win the unique constraint.
    contact = Contact(owner_id=owner.id, contact_id=peer.id)
    db.add(contact)
    try:
        await db.commit()
    except IntegrityError:
        # Another request created the same edge first; load that winner instead of 500ing.
        await db.rollback()
        existing = await db.scalar(
            select(Contact).where(Contact.owner_id == owner.id, Contact.contact_id == peer.id)
        )
        if existing is None:
            # The unique-constraint race should always leave a row; fail closed if not.
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="could not save contact",
            ) from None
        return serialize_contact(existing, peer)

    await db.refresh(contact)
    return serialize_contact(contact, peer)


# Remove a named account from the owner's address book, idempotently.
async def remove_contact_for_owner(db: AsyncSession, owner: User, username: str) -> None:
    """Delete (owner_id, contact_id) if present; never errors on "not a contact."

    The legacy React prototype never implemented contact delete at all (its
    address book was add-only); removal here matches the existing add
    endpoint's idempotent shape rather than 404ing on an already-absent row.
    """

    peer = await db.scalar(select(User).where(User.username == username))
    if peer is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_USER_NOT_FOUND_DETAIL)

    existing = await db.scalar(
        select(Contact).where(Contact.owner_id == owner.id, Contact.contact_id == peer.id)
    )
    if existing is not None:
        await db.delete(existing)
        await db.commit()
