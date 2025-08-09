"""Lightweight Alembic migrations runner for programmatic use."""
import os
import logging
from typing import Optional

from alembic import command
from alembic.config import Config as AlembicConfig

logger = logging.getLogger(__name__)


def _build_alembic_config(database_url: str) -> AlembicConfig:
    """Create an Alembic Config pointing to our migrations directory.

    Works with or without an alembic.ini on disk (falls back to in-memory config).
    """
    # Resolve paths relative to project root (two levels up from this file: src/database -> project root)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ini_path = os.path.join(base_dir, "alembic.ini")
    scripts_path = os.path.join(base_dir, "src", "database", "migrations")

    if os.path.exists(ini_path):
        cfg = AlembicConfig(ini_path)
    else:
        # Fallback: build config in-memory so we don't require alembic.ini to exist in the container
        cfg = AlembicConfig()
    cfg.set_main_option("script_location", scripts_path)

    # Provide DB URL for Alembic; env.py also supports reading from env/config
    if database_url:
        cfg.set_main_option("sqlalchemy.url", database_url)
        os.environ.setdefault("DATABASE_URL", database_url)

    return cfg


def run_migrations(database_url: str, target: str = "head") -> None:
    """Upgrade database to target revision (default: head)."""
    try:
        logger.info(f"Running Alembic migrations -> {target}")
        cfg = _build_alembic_config(database_url)
        command.upgrade(cfg, target)
        logger.info("✅ Alembic migrations completed")
    except Exception as e:
        logger.error(f"❌ Alembic migrations failed: {e}")
        raise
