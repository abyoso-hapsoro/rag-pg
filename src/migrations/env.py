# Environment imports
import os
from dotenv import load_dotenv
load_dotenv()

# Supplementary import
from logging.config import fileConfig

# Core imports
from alembic import context
from sqlalchemy import create_engine
from src.models.document import Base

# Retrieve database URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required for Alembic migrations.")

# Access values within associated .ini file
config = context.config

# Interpret the config file for logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Specify metadata (models mapped)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode — not creating or requiring a live database Engine.
    SQL emitted by context.execute() will be written to stdout or the script output.
    """

    # Base kwargs for Alembic context configuration
    kwargs = {'url': DATABASE_URL, 'literal_bind': True}
    
    # Provide metadata so Alembic knows the schema state, if available
    if target_metadata:
        kwargs['target_metadata'] = target_metadata
    
    # Configure Alembic context for offline execution
    context.configure(**kwargs)

    # Perform virtual transaction for emitting SQL statements
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode — creating a database Engine which connects to PostgreSQL instance.
    Migrations are applied directly against the live database.
    """

    # Create an SQLAlchemy Engine
    engine = create_engine(DATABASE_URL)
    
    # Open database session-level connection
    with engine.connect() as conn:
        # Base kwargs for Alembic context configuration
        kwargs = {'connection': conn}

        # Provide metadata so Alembic knows the schema state, if available
        if target_metadata:
            kwargs['target_metadata'] = target_metadata

        # Configure Alembic to use the active database connection
        context.configure(**kwargs)

        # Perform transaction and apply migrations to database
        with context.begin_transaction():
            context.run_migrations()


# Determine action based on context
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
