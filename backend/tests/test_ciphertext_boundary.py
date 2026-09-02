"""Grep-sweep: the server must never handle private keys or plaintext message bodies."""

# Import the AST walker used to inspect identifiers in application source.
import ast

# Import Path so the sweep walks backend/app relative to this test file.
from pathlib import Path

# Import every model so Base.metadata is fully populated before the column sweep.
from app import models  # noqa: F401

# Import the declarative metadata that lists every mapped column name.
from app.db import Base

# Identifiers that would mean the server is doing client-only crypto or storing plaintext.
_FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "private_key",
        "privateKey",
        "plaintext",
        "message_body",
        "decrypt",
        "decrypt_message",
        "crypto_aead_xchacha20poly1305_ietf_decrypt",
        "crypto_secretbox",
        "crypto_secretbox_open_easy",
        "session_key",
        "epoch_key",
        "master_key",
    }
)

# Column names that must never appear on any mapped table.
_FORBIDDEN_COLUMNS = frozenset(
    {
        "plaintext",
        "private_key",
        "session_key",
        "epoch_key",
        "master_key",
        "message_body",
        "body",
        "content",
        "text",
    }
)


# Confirm the messages table only stores envelope fields, never a body or key.
def test_messages_table_stores_only_ciphertext_nonce_and_epoch() -> None:
    """Require the messages table's columns to match spec §5, plus pre-deployment additions.

    ad_version/client_message_id/revision/edited_at were added to support safe
    message editing (a client-generated message identity plus a revision
    counter bound into the v2 associated data). None of the four is plaintext
    or key material: ad_version and revision are small integers,
    client_message_id is a UUID the client already chose before encrypting,
    and edited_at is only a timestamp.
    """

    columns = set(Base.metadata.tables["messages"].c.keys())
    assert columns == {
        "id",
        "conversation_id",
        "sender_id",
        "ciphertext",
        "nonce",
        "key_epoch",
        "created_at",
        "ad_version",
        "client_message_id",
        "revision",
        "edited_at",
    }


# Confirm conversation_reads stores only a last-read cursor, never a preview string.
def test_conversation_reads_table_stores_cursor_only() -> None:
    """Require (user_id, conversation_id, last_read_at) metadata, never a body."""

    columns = set(Base.metadata.tables["conversation_reads"].c.keys())
    assert columns == {
        "id",
        "user_id",
        "conversation_id",
        "last_read_at",
        "last_read_message_id",
    }


# Confirm message_receipts stores only delivered/read timestamps.
def test_message_receipts_table_stores_ticks_only() -> None:
    """Require (message_id, recipient_id, delivered_at, read_at) only."""

    columns = set(Base.metadata.tables["message_receipts"].c.keys())
    assert columns == {
        "id",
        "message_id",
        "recipient_id",
        "delivered_at",
        "read_at",
    }


# Confirm encrypted_blobs stores sealed bytes, never opened file pixels.
def test_encrypted_blobs_table_stores_ciphertext_only() -> None:
    """Require ciphertext, nonce, and routing metadata, never a body column."""

    columns = set(Base.metadata.tables["encrypted_blobs"].c.keys())
    assert columns == {
        "id",
        "conversation_id",
        "uploader_id",
        "ciphertext",
        "nonce",
        "byte_length",
        "created_at",
    }


# Confirm users gained public profile columns without growing a body or key column.
def test_users_table_public_profile_columns_are_not_message_bodies() -> None:
    """Require avatar_bytes/display_name/bio as public metadata, never a chat body."""

    columns = set(Base.metadata.tables["users"].c.keys())
    assert "display_name" in columns
    assert "bio" in columns
    assert "avatar_bytes" in columns
    assert "avatar_media_type" in columns
    assert "content" not in columns
    assert "body" not in columns
    assert "text" not in columns
    assert "plaintext" not in columns


# Confirm blocks store only who-blocked-whom metadata, never a message body or key.
def test_blocks_table_stores_blocker_and_blocked_ids_only() -> None:
    """Require (blocker_id, blocked_id) metadata, matching the pre-deployment review."""

    columns = set(Base.metadata.tables["blocks"].c.keys())
    assert columns == {
        "id",
        "blocker_id",
        "blocked_id",
        "created_at",
    }


# Confirm reports store only metadata and a bounded free-text reason, never message content.
def test_reports_table_stores_metadata_only() -> None:
    """Require (reporter_id, reported_id, reason, created_at) only.

    There is deliberately no message-content or message-id column: the
    server cannot read message ciphertext, so it cannot attach reported
    message text without breaking the E2EE trust boundary.
    """

    columns = set(Base.metadata.tables["reports"].c.keys())
    assert columns == {
        "id",
        "reporter_id",
        "reported_id",
        "reason",
        "created_at",
    }


# Confirm message_hides stores only a per-owner hide marker, never a message body.
def test_message_hides_table_stores_owner_and_message_ids_only() -> None:
    """Require (owner_id, message_id, created_at) only, matching the pre-deployment review."""

    columns = set(Base.metadata.tables["message_hides"].c.keys())
    assert columns == {
        "id",
        "owner_id",
        "message_id",
        "created_at",
    }


# Confirm conversations gained last_rotated_at without growing a key column.
def test_conversations_table_stores_epoch_counter_not_keys() -> None:
    """Require membership + current_epoch + last_rotated_at, never a session key.

    messages_in_current_epoch is an integer counter used so epoch rotation does
    not COUNT(*) the messages table on every persist. It is not key material.
    """

    # Read the mapped conversations columns after Slice 8's rotation timestamp.
    columns = set(Base.metadata.tables["conversations"].c.keys())
    # last_rotated_at is a timestamp used by the 24h rule, not key material.
    assert columns == {
        "id",
        "user_a_id",
        "user_b_id",
        "current_epoch",
        "last_rotated_at",
        "messages_in_current_epoch",
        "created_at",
    }


# Confirm contacts store address-book edges only.
def test_contacts_table_stores_owner_and_contact_ids_only() -> None:
    """Require (owner_id, contact_id) metadata, never a message body or score."""

    # Read the mapped contacts columns added in Slice 7.
    columns = set(Base.metadata.tables["contacts"].c.keys())
    # No plaintext, score, or key column may appear here.
    assert columns == {
        "id",
        "owner_id",
        "contact_id",
        "created_at",
    }


# Confirm no mapped table grew a plaintext or private-key column.
def test_schema_has_no_plaintext_or_private_key_columns() -> None:
    """Fail if any ORM table exposes a forbidden secret or body column name."""

    for table in Base.metadata.tables.values():
        forbidden = _FORBIDDEN_COLUMNS.intersection(table.c.keys())
        assert not forbidden, f"{table.name} has forbidden columns: {forbidden}"


# Confirm application Python never names decrypt/private-key/plaintext identifiers.
def test_app_source_never_handles_private_keys_or_plaintext() -> None:
    """Walk backend/app/*.py ASTs and reject forbidden crypto/plaintext identifiers.

    Comments are invisible to the AST, so documentation that says 'never store
    plaintext' is allowed. Using those names as variables, functions, or
    attributes in executable code is not.
    """

    app_root = Path(__file__).resolve().parents[1] / "app"
    offenders: list[str] = []
    for path in app_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name: str | None = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = node.name
            if name in _FORBIDDEN_IDENTIFIERS:
                line = getattr(node, "lineno", 0)
                offenders.append(f"{path.relative_to(app_root.parent)}:{line}:{name}")
    assert offenders == []


# Confirm every Message SELECT in application code is conversation-scoped (§2, §11).
def test_message_selects_in_app_are_conversation_scoped() -> None:
    """Fail if app code selects Message rows without filtering conversation_id.

    Tests may load the whole table for assertions. Application code must not
    expose a flat all-messages query with client-side filtering.
    """

    # Walk only runtime application modules, not tests.
    app_root = Path(__file__).resolve().parents[1] / "app"
    # Collect files that select Message rows without a conversation_id filter.
    offenders: list[str] = []
    # Inspect every Python module under backend/app.
    for path in app_root.rglob("*.py"):
        # Read source as text so a missing filter is a cheap substring check.
        text = path.read_text(encoding="utf-8")
        # Detect SQLAlchemy selects against the messages mapped class.
        selects_messages = "select(Message)" in text or "select_from(Message)" in text
        # Skip files that never query the messages table.
        if not selects_messages:
            # This module does not select envelopes.
            continue
        # Require the same file to filter on conversation_id (never a global dump).
        if "Message.conversation_id" not in text:
            # Record the relative path for the assertion message.
            offenders.append(str(path.relative_to(app_root.parent)))
    # Application Message reads must stay conversation-scoped.
    assert offenders == []


# Confirm no HTTP route exposes an unscoped /messages collection.
def test_no_flat_all_messages_http_route() -> None:
    """Require every messages HTTP path to include {conversation_id}."""

    # Import the live app so published paths match what Compose will serve.
    from app.main import app

    # Use the OpenAPI map: FastAPI 0.116 nests included routers, so app.routes
    # has no path string for GET /conversations/{conversation_id}/messages.
    published_paths = list(app.openapi()["paths"])
    # Collect paths that mention messages (history GET is the only one today).
    message_paths = [path for path in published_paths if "message" in path.lower()]
    # Spec §11 forbids GET /messages with client-side filtering.
    for path in message_paths:
        # History must stay conversation-scoped.
        assert "{conversation_id}" in path, path
    # History GET must exist so this check cannot pass by having zero message routes.
    assert "/conversations/{conversation_id}/messages" in message_paths


# Confirm the Docker entrypoint is not imported by application code (pytest must not migrate).
def test_app_modules_do_not_import_docker_entrypoint() -> None:
    """Keep migrate-on-start in docker-entrypoint.py, outside the ASGI import graph."""

    # Walk only runtime application modules.
    app_root = Path(__file__).resolve().parents[1] / "app"
    # Collect any module that would pull Alembic into pytest.
    offenders: list[str] = []
    # Inspect every Python file the API process imports.
    for path in app_root.rglob("*.py"):
        # Read source as text for a cheap substring check.
        text = path.read_text(encoding="utf-8")
        # Fail if a router or service starts the container entrypoint.
        if "docker-entrypoint" in text:
            # Record the relative path for the assertion message.
            offenders.append(str(path.relative_to(app_root.parent)))
    # Application imports must stay free of the Compose startup script.
    assert offenders == []
