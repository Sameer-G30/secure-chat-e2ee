"""Validate API configuration loaded exclusively from environment variables."""

# Cache the settings object so environment parsing occurs once per process.
from functools import lru_cache

# Import Pydantic's settings base class and settings-specific options.
from pydantic_settings import BaseSettings, SettingsConfigDict


# Define every configuration value the first backend slice consumes.
class Settings(BaseSettings):
    """Represent validated runtime settings for the API."""

    # Identify the current deployment environment in diagnostics.
    app_env: str = "development"
    # Select the interface on which the ASGI server listens.
    api_host: str = "0.0.0.0"
    # Select the TCP port on which the ASGI server listens.
    api_port: int = 8000
    # Provide the async PostgreSQL connection URL for later database slices.
    database_url: str = "postgresql+asyncpg://secure_chat:change-me@postgres:5432/secure_chat"
    # Define the browser origin that future CORS policy will permit.
    frontend_origin: str = "http://localhost:5173"
    # Sign and verify JWTs with a secret that must be overridden in every real deployment.
    jwt_secret_key: str = "replace-with-at-least-32-random-bytes"
    # Pin the signing algorithm explicitly rather than trusting a token-supplied header.
    jwt_algorithm: str = "HS256"
    # Keep access tokens short-lived so a leaked token has a small blast radius.
    access_token_expire_minutes: int = 15
    # Let refresh tokens live far longer, since rotation-on-use limits replay risk.
    refresh_token_expire_days: int = 7
    # Bump conversations.current_epoch after this many persisted envelopes since the last bump.
    #
    # Spec §6.7 / Slice 8. Default 50 so a quiet chat does not rotate every send.
    # Set to 0 to disable the message-count trigger (time trigger can still fire).
    # Tests and the two-tab demo set this to 2 via EPOCH_ROTATE_AFTER_MESSAGES.
    epoch_rotate_after_messages: int = 50
    # Also bump when this many hours have passed since last_rotated_at or created_at.
    #
    # Combined with the message-count rule by OR: either trigger is enough.
    # Set to 0 to disable the wall-clock trigger (message-count can still fire).
    epoch_rotate_after_hours: int = 24
    # Read case-insensitive variables without silently accepting unknown keys.
    model_config = SettingsConfigDict(
        # Load a local untracked file when the API runs outside Docker.
        env_file="../.env",
        # Interpret environment variable names without case sensitivity.
        case_sensitive=False,
        # Reject misspelled values found in the local environment file.
        extra="ignore",
    )


# Memoize configuration while retaining an overrideable dependency function.
@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""

    # Construct the settings from defaults, .env, and process environment.
    return Settings()
