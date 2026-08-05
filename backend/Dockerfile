# Pin the exact Python patch version selected for the project.
FROM python:3.11.9-slim-bookworm

# Copy the pinned uv binary without installing another package manager.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# Prevent Python from buffering container logs.
ENV PYTHONUNBUFFERED=1
# Compile bytecode during dependency installation for faster startup.
ENV UV_COMPILE_BYTECODE=1
# Copy packages when Docker cache and environment filesystems differ.
ENV UV_LINK_MODE=copy

# Keep all backend files under one predictable container directory.
WORKDIR /app

# Copy dependency metadata before source code to preserve Docker cache layers.
COPY pyproject.toml uv.lock ./
# Install only production dependencies into the container virtual environment.
RUN uv sync --frozen --no-dev

# Copy the API package after dependencies have been cached.
COPY app ./app
# Copy migration configuration so future slices can migrate in this image.
COPY alembic.ini ./
# Copy the initialized Alembic environment for schema management.
COPY alembic ./alembic

# Document the HTTP port served by the ASGI process.
EXPOSE 8000

# Start the API through uv without modifying the locked environment.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
