"""Database connection management for BSE Predict."""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging
from typing import Optional

logger = logging.getLogger(__name__)

Base = declarative_base()


class DatabaseConnection:
    """Manages database connections with connection pooling."""
    
    def __init__(self, database_url: str):
        """Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        
    def test_connection(self) -> bool:
        """Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            return False
            
    def get_session(self):
        """Get database session.
        
        Returns:
            Database session
        """
        return self.SessionLocal()
        
    def execute_raw_sql(self, query: str, params: Optional[dict] = None):
        """Execute raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query result
        """
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return result.fetchall()
