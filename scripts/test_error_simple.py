#!/usr/bin/env python3
"""Simple test for error logging."""
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# First initialize the app properly to enable DB logging
from src.utils.config import config
from src.database.operations import db_ops
from src.utils.logger import setup_logging, get_logger

# Enable database error logging
setup_logging(use_db_handler=True)
logger = get_logger(__name__)

print("Testing error logging...")

# Log some errors
logger.error("Test error 1: Simple error")
logger.critical("Test error 2: Critical error")

try:
    x = 1 / 0
except Exception as e:
    logger.error("Test error 3: Division by zero", exc_info=True)

# Force flush of async handler
from src.utils.db_logger import AsyncDatabaseLogHandler
for handler in logger.handlers:
    if isinstance(handler, AsyncDatabaseLogHandler):
        print("Flushing async handler...")
        handler.flush()
        time.sleep(1)  # Give it time to complete

print("\nChecking database...")

# Check errors
with db_ops.db.get_session() as session:
    from sqlalchemy import text
    result = session.execute(text("SELECT COUNT(*) FROM system_errors"))
    count = result.scalar()
    print(f"Total errors in database: {count}")
    
    # Get recent errors
    result = session.execute(text("""
        SELECT timestamp, level, message 
        FROM system_errors 
        ORDER BY timestamp DESC 
        LIMIT 5
    """))
    rows = result.fetchall()
    
    if rows:
        print("\nRecent errors:")
        for row in rows:
            print(f"  [{row[0]}] {row[1]}: {row[2][:50]}...")
    else:
        print("\nNo errors found in database")