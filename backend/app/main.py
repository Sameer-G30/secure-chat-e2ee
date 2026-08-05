"""Create the ciphertext-only FastAPI service."""

# Import Annotated to describe dependency-injected parameter metadata.
from typing import Annotated

# Import FastAPI's application and dependency injection primitives.
from fastapi import Depends, FastAPI

# Import the validated settings model and cached provider.
from app.config import Settings, get_settings

# Construct the ASGI application with explicit public metadata.
app = FastAPI(
    # Display the project name in generated API documentation.
    title="Secure Chat API",
    # Describe the server's ciphertext-only trust boundary.
    description="Stores and relays encrypted message envelopes without plaintext access.",
    # Identify the first vertical-slice API version.
    version="0.1.0",
)


# Expose a lightweight endpoint for local checks and container health probes.
@app.get("/health", tags=["operations"])
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
    """Report whether the API process is ready to receive requests."""

    # Return only non-sensitive operational metadata.
    return {
        # Confirm that the HTTP process is healthy.
        "status": "ok",
        # Identify which deployment profile supplied configuration.
        "environment": settings.app_env,
        # Expose the running API version for smoke-test assertions.
        "version": app.version,
    }
