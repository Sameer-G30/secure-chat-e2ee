"""Issue short-lived, single-use tickets so access tokens never appear in WebSocket URLs."""

# Import secrets for unguessable ticket strings and time for expiry checks.
import secrets
import time
from uuid import UUID

# Keep tickets valid only briefly: the browser connects immediately after requesting one.
TICKET_TTL_SECONDS = 30.0
# Bound memory use if a client requests tickets without ever connecting.
_MAX_PENDING_TICKETS = 10_000

# Map ticket -> (user_id, conversation_id, expiry monotonic time). Process-local like ConnectionHub.
_tickets: dict[str, tuple[UUID, UUID, float]] = {}


def _purge_expired(now: float) -> None:
    """Drop expired tickets so the dict cannot grow without bound."""

    # Collect first so the dict is not mutated while iterating.
    expired = [key for key, (_, _, expiry) in _tickets.items() if expiry <= now]
    for key in expired:
        del _tickets[key]


def issue_ticket(user_id: UUID, conversation_id: UUID) -> str:
    """Create a one-time ticket bound to one user and one conversation."""

    now = time.monotonic()
    _purge_expired(now)
    # Refuse rather than evict legitimate tickets when the table is saturated.
    if len(_tickets) >= _MAX_PENDING_TICKETS:
        raise RuntimeError("too many pending websocket tickets")
    # 32 random bytes encoded URL-safe is unguessable.
    ticket = secrets.token_urlsafe(32)
    _tickets[ticket] = (user_id, conversation_id, now + TICKET_TTL_SECONDS)
    return ticket


def redeem_ticket(ticket: str, conversation_id: UUID) -> UUID | None:
    """Consume a ticket and return its user id, or None if invalid, expired, or mismatched."""

    # pop makes the ticket single-use even when validation then fails.
    entry = _tickets.pop(ticket, None)
    if entry is None:
        return None
    user_id, bound_conversation, expiry = entry
    # Reject expired tickets and tickets presented for another conversation.
    if expiry <= time.monotonic() or bound_conversation != conversation_id:
        return None
    return user_id
