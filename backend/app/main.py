"""Create the ciphertext-only FastAPI service."""

# Import Annotated to describe dependency-injected parameter metadata.
from typing import Annotated

# Import FastAPI's application and dependency injection primitives.
from fastapi import Depends, FastAPI

# Import CORS middleware so the Vite dev origin may call authenticated endpoints.
from fastapi.middleware.cors import CORSMiddleware

# Import slowapi's rate-limit exception and the response handler that renders it.
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Import the validated settings model and cached provider.
from app.config import Settings, get_settings

# Import the authentication router added in Slice 2 and extended in Slice 3,
# the key upload/lookup router added in Slice 3 (plus the §6.4 epoch alias),
# and the Slice 4 conversation REST + WebSocket ciphertext-relay routers.
from app.routers import auth, contacts, conversations, keys, ws

# Import the shared limiter instance so the app enforces the same rate limits.
from app.security.rate_limit import limiter

# Construct the ASGI application with explicit public metadata.
app = FastAPI(
    # Display the project name in generated API documentation.
    title="Secure Chat API",
    # Describe the server's ciphertext-only trust boundary.
    description="Stores and relays encrypted message envelopes without plaintext access.",
    # Identify the ninth vertical-slice API version.
    version="0.9.0",
)

# Attach the limiter so every @limiter.limit(...) decorator can read shared state.
app.state.limiter = limiter
# Register slowapi's handler so exceeding a limit returns a clean 429, not a 500.
# slowapi's handler signature is narrower than Starlette's generic Exception handler
# type, which mypy correctly flags; the runtime contract is exactly what Starlette expects.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Allow only the configured frontend origin to call the API with credentials.
app.add_middleware(
    CORSMiddleware,
    # Read the permitted browser origin from validated settings, not a hardcoded value.
    allow_origins=[get_settings().frontend_origin],
    # Permit the Authorization header carrying Slice 3's bearer access tokens.
    allow_credentials=True,
    # Allow the HTTP verbs this API's routers currently use.
    allow_methods=["GET", "POST"],
    # Allow the headers the frontend needs to send, including future auth headers.
    allow_headers=["Authorization", "Content-Type"],
)

# Mount the authentication router's endpoints onto the application.
app.include_router(auth.router)
# Mount the key upload/lookup router (and the spec §6.4 epoch alias).
app.include_router(keys.router)
# Mount 1:1 conversation create/fetch and the REST epoch endpoint.
app.include_router(conversations.router)
# Mount the authenticated server-side contact address book.
app.include_router(contacts.router)
# Mount the authenticated ciphertext-only WebSocket relay.
app.include_router(ws.router)


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
