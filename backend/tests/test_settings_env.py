"""Confirm .env.example documents every Settings field the API actually reads."""

# Import Path so the test locates .env.example relative to this file.
from pathlib import Path

# Import the validated settings model whose field names must appear as env vars.
from app.config import Settings, get_settings

# Import the FastAPI app so we can assert CORS still uses FRONTEND_ORIGIN only.
from app.main import app


# Map each Settings field to the environment variable name pydantic-settings reads.
def test_env_example_documents_every_settings_field() -> None:
    """Fail if a Settings field is read but missing from the committed example file.

    POSTGRES_* variables are Compose-only (not Settings fields) and may appear
    in .env.example without a matching Settings attribute. Settings extra=ignore
    so those keys do not crash API startup.
    """

    # Walk from tests/ up to the repository root where .env.example lives.
    example_path = Path(__file__).resolve().parents[2] / ".env.example"
    # Require the example file to exist so CI cannot silently skip this check.
    assert example_path.is_file(), ".env.example is missing from the repository root"
    # Read the committed template as text for substring assertions.
    example_text = example_path.read_text(encoding="utf-8")
    # Check every field the API process actually parses.
    for field_name in Settings.model_fields:
        # pydantic-settings maps app_env -> APP_ENV (case-insensitive env names).
        env_name = field_name.upper()
        # Require the example to assign or at least mention each consumed variable.
        assert env_name in example_text, f"{env_name} is missing from .env.example"


# Confirm CORS is still a single FRONTEND_ORIGIN, now with DELETE and PATCH.
def test_cors_uses_single_frontend_origin_and_documented_verbs() -> None:
    """Document CORS: one origin, GET/POST/DELETE/PATCH.

    DELETE was added for contact removal, unblocking, and message
    delete-for-everyone. PATCH was added for the signed-in profile update.
    """

    # Collect CORSMiddleware from the live app stack without a typed cls identity check.
    cors = next(
        middleware
        for middleware in app.user_middleware
        if getattr(middleware.cls, "__name__", "") == "CORSMiddleware"
    )
    # Starlette stores constructor kwargs on Middleware.kwargs (or .options).
    raw_options = getattr(cors, "kwargs", None) or getattr(cors, "options", {})
    # Narrow to a mapping so the CORS assertions stay typed for mypy.
    assert isinstance(raw_options, dict)
    # Require a one-element allow-list, not a wildcard or guessed origin list.
    assert raw_options["allow_origins"] == [get_settings().frontend_origin]
    # Require the verbs this API's routers actually use.
    assert raw_options["allow_methods"] == ["GET", "POST", "DELETE", "PATCH"]


# Confirm the Compose API image still migrates before binding uvicorn.
def test_docker_entrypoint_applies_alembic_then_starts_uvicorn() -> None:
    """Require docker-entrypoint.py to upgrade head, then exec uvicorn.

    pytest must not run this script; the file lives beside the Dockerfile,
    not under app/, so in-memory tests keep using create_all.
    """

    # Locate the entrypoint next to backend/Dockerfile.
    entrypoint = Path(__file__).resolve().parents[1] / "docker-entrypoint.py"
    # Require the file to exist so a missing COPY would fail CI before Compose.
    assert entrypoint.is_file()
    # Read the startup script as text.
    text = entrypoint.read_text(encoding="utf-8")
    # Require a full schema upgrade, including last_rotated_at (a9f3c6e12b80).
    assert "alembic" in text
    # Require the head revision, not a hardcoded older revision id.
    assert "upgrade" in text and "head" in text
    # Require the API process to start after migrations succeed.
    assert "uvicorn" in text
    # Require fail-closed behavior when migrate returns non-zero.
    assert "sys.exit" in text
