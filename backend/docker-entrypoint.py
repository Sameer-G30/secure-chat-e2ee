"""Apply pending Alembic revisions, then start the ciphertext-only API.

Slice 9: `docker compose up` must reach a migrated schema without a second
manual `alembic upgrade head` command. pytest does not import or run this
file; tests keep using in-memory SQLite plus Base.metadata.create_all.
"""

# Import os so this process can be replaced by uvicorn after a successful migrate.
import os

# Import subprocess to run Alembic before the ASGI server binds the port.
import subprocess

# Import sys to exit non-zero when schema application fails.
import sys


# Apply schema, then exec uvicorn so container stop signals reach the server.
def main() -> None:
    """Run `alembic upgrade head`, then replace this process with uvicorn."""

    # Apply every pending revision, including last_rotated_at (a9f3c6e12b80).
    migrate = subprocess.run(
        # Reuse the image's locked environment; do not sync on every container start.
        ["uv", "run", "--no-sync", "alembic", "upgrade", "head"],
        # Inherit stdout/stderr so Compose logs show migration progress.
        check=False,
    )
    # Fail the container if schema application did not succeed.
    if migrate.returncode != 0:
        # Leave uvicorn unstarted so /health cannot report ready on an unmigrated DB.
        sys.exit(migrate.returncode)
    # Replace this process with uvicorn; argv[0] must be the executable name.
    os.execvp(
        # Look up uv on PATH inside the python:3.11.9-slim-bookworm image.
        "uv",
        # Match the previous Dockerfile CMD arguments exactly.
        [
            "uv",
            "run",
            "--no-sync",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    )


# Run only when the image starts this file as the container entrypoint.
if __name__ == "__main__":
    # Start migrations then the API.
    main()
