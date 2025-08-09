import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Ensure project root is on sys.path so we can import project modules when running CLI
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def get_database_url() -> str:
    # Prefer environment variable, otherwise fall back to project config
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    try:
        from src.utils.config import Config
        return Config().database_url
    except Exception:
        raise RuntimeError(
            "DATABASE_URL is not set and Config() could not be loaded. "
            "Set DATABASE_URL env var before running migrations."
        )


def run_migrations_offline():
    url = get_database_url()
    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    url = get_database_url()

    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None,  # No ORM models metadata; we use SQL/ops only
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
