"""Database initialization orchestrated via Alembic migrations."""
from sqlalchemy import text
from src.database.connection import DatabaseConnection
from src.utils.logger import get_logger
from src.database.migrations_runner import run_migrations

logger = get_logger(__name__)


class DatabaseInitializer:
    """Runs database migrations and verifies required tables exist."""
    
    def __init__(self, database_url: str):
        self.db = DatabaseConnection(database_url)
    
    def verify_tables_exist(self) -> dict:
        """Verify all required tables exist."""
        required_tables = [
            "ohlcv_data",
            "feature_cache",
            "futures_data",
            "predictions",
            "prediction_outcomes",
            "model_performance",
            "telegram_reports",
            "system_health",
            "system_errors",
            "model_training_history",
        ]

        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE';
        """

        try:
            with self.db.engine.connect() as conn:
                result = conn.execute(text(query))
                existing_tables = [row[0] for row in result]

                status = {table: (table in existing_tables) for table in required_tables}
                missing = [t for t, exists in status.items() if not exists]
                if missing:
                    logger.warning(f"Missing tables: {missing}")
                else:
                    logger.info("✅ All required tables exist")

                return status

        except Exception as e:
            logger.error(f"Error verifying tables: {e}")
            return {}

    def initialize_all_tables(self):
        """Initialize database schema using Alembic migrations (idempotent)."""
        logger.info("Initializing database schema via Alembic migrations...")

        # Ensure database is reachable before running migrations
        if not self.db.test_connection():
            raise RuntimeError("Database connection failed; cannot run migrations")

        # Run migrations to head
        run_migrations(self.db.database_url, "head")

        # Verify all tables exist
        return self.verify_tables_exist()


def init_database(database_url: str):
    """Initialize database with all required tables via Alembic migrations."""
    initializer = DatabaseInitializer(database_url)
    return initializer.initialize_all_tables()
