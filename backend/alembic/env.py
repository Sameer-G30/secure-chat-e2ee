"""Configure asynchronous Alembic migrations from validated app settings."""

# Import asyncio to execute Alembic's asynchronous migration coroutine.
import asyncio

# Import logging configuration support used by Alembic's INI file.
from logging.config import fileConfig

# Import SQLAlchemy connection and pool types used by migration runners.
from sqlalchemy import Connection, pool

# Import SQLAlchemy's asynchronous engine factory.
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import Alembic's process-wide migration context.
from alembic import context

# Import the same validated database URL used by the API.
from app.config import get_settings

# Access the active Alembic configuration object.
config = context.config
# Override placeholder INI credentials with environment-backed settings.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

# Configure migration loggers when Alembic provides an INI filename.
if config.config_file_name is not None:
    # Apply the logger, handler, and formatter sections from alembic.ini.
    fileConfig(config.config_file_name)

# Use no schema metadata until SQLAlchemy models arrive in Slice 2.
target_metadata = None


# Generate migration SQL without opening a database connection.
def run_migrations_offline() -> None:
    """Run migrations in offline SQL-rendering mode."""

    # Read the environment-backed URL assigned above.
    url = config.get_main_option("sqlalchemy.url")
    # Configure Alembic to emit portable named SQL parameters.
    context.configure(
        # Select the target database dialect from the configured URL.
        url=url,
        # Supply model metadata once models exist in a later slice.
        target_metadata=target_metadata,
        # Render literal values into generated offline SQL.
        literal_binds=True,
        # Use named placeholders in generated SQL.
        dialect_opts={"paramstyle": "named"},
    )

    # Group all migration operations in one transaction.
    with context.begin_transaction():
        # Execute pending migration directives.
        context.run_migrations()


# Apply migrations using a live synchronous connection adapter.
def do_run_migrations(connection: Connection) -> None:
    """Run configured migrations through an adapted async connection."""

    # Bind Alembic's migration context to the active connection.
    context.configure(connection=connection, target_metadata=target_metadata)

    # Group all live migration operations in one transaction.
    with context.begin_transaction():
        # Execute pending migration directives.
        context.run_migrations()


# Open and dispose the asynchronous SQLAlchemy migration engine.
async def run_async_migrations() -> None:
    """Create an async engine and execute live migrations."""

    # Build an uncached engine from Alembic's configured SQLAlchemy section.
    connectable = async_engine_from_config(
        # Read all active Alembic INI options.
        config.get_section(config.config_ini_section, {}),
        # Select options beginning with sqlalchemy.
        prefix="sqlalchemy.",
        # Avoid retaining migration connections in a pool.
        poolclass=pool.NullPool,
    )

    # Open one asynchronous database connection.
    async with connectable.connect() as connection:
        # Adapt the connection for Alembic's synchronous migration API.
        await connection.run_sync(do_run_migrations)

    # Release engine resources after migration completion.
    await connectable.dispose()


# Select offline rendering or online database execution.
if context.is_offline_mode():
    # Generate SQL without connecting to PostgreSQL.
    run_migrations_offline()
else:
    # Run the async migration coroutine to completion.
    asyncio.run(run_async_migrations())
