"""Validate receipt and last-read payloads that never include a message body."""

# Import datetime for the last-read response timestamp.
from datetime import datetime

# Import UUID for typed conversation and envelope identifiers.
from uuid import UUID

# Import Pydantic's model base and field bounds.
from pydantic import BaseModel, Field

# Cap how many envelope ids one WebSocket receipt frame may acknowledge.
MAX_RECEIPT_BATCH = 100


# Describe one WebSocket receipt frame a connected client may send.
class ReceiptFrameIn(BaseModel):
    """Represent delivered/read acknowledgements for envelopes this member received.

    The server never decrypts those envelopes. kind is metadata only.
    """

    # Mark this frame as a receipt so the WebSocket loop can distinguish it from ciphertext.
    type: str = "receipt"
    # Distinguish a device-ack (delivered) from a focused-chat ack (read).
    kind: str = Field(pattern="^(delivered|read)$")
    # Identify the stored envelopes this recipient is acknowledging.
    message_ids: list[UUID] = Field(min_length=1, max_length=MAX_RECEIPT_BATCH)


# Describe the last-read cursor returned after POST /conversations/{id}/read.
class ConversationReadResponse(BaseModel):
    """Represent one member's last-read cursor without a preview string."""

    # Identify the conversation this cursor belongs to.
    conversation_id: UUID
    # Record when this member last marked the conversation read.
    last_read_at: datetime
