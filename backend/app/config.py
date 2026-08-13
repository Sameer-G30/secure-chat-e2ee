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
