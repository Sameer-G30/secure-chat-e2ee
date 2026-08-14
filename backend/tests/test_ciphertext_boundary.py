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
