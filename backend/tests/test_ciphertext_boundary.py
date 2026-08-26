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
    """Require the messages table's columns to match spec §5 exactly."""

    columns = set(Base.metadata.tables["messages"].c.keys())
    assert columns == {
        "id",
        "conversation_id",
        "sender_id",
        "ciphertext",
        "nonce",
        "key_epoch",
        "created_at",
    }


# Confirm conversations gained last_rotated_at without growing a key column.
def test_conversations_table_stores_epoch_counter_not_keys() -> None:
    """Require membership + current_epoch + last_rotated_at, never a session key."""

    # Read the mapped conversations columns after Slice 8's rotation timestamp.
    columns = set(Base.metadata.tables["conversations"].c.keys())
    # last_rotated_at is a timestamp used by the 24h rule, not key material.
    assert columns == {
        "id",
        "user_a_id",
        "user_b_id",
        "current_epoch",
        "last_rotated_at",
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
