#!/usr/bin/env python3
"""Test script to verify error logging to database works."""

import sys
import os
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import get_logger
from src.utils.db_logger import get_db_logger_stats
from src.database.operations import db_ops

logger = get_logger(__name__)


def test_error_logging():
    """Test various error logging scenarios."""
    print("🧪 Testing error logging to database...")
    print("-" * 50)
    
    # Test 1: Simple error
    print("\n1. Testing simple error logging...")
    logger.error("Test error message - this is a test error")
    
    # Test 2: Error with exception
    print("\n2. Testing error with exception...")
    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error("Division by zero error occurred", exc_info=True)
    
    # Test 3: Critical error
    print("\n3. Testing critical error...")
    logger.critical("Test critical error - system malfunction simulation")
    
    # Test 4: Error from different module
    print("\n4. Testing error from different module...")
    test_logger = get_logger("src.models.test_module")
    test_logger.error("Error from model training module")
    
    # Give async handler time to flush
    print("\n⏳ Waiting for async handler to flush...")
    time.sleep(6)
    
    # Check if errors were logged
    print("\n📊 Checking database for logged errors...")
    print("-" * 50)
    
    try:
        # Get error statistics
        stats = get_db_logger_stats()
        
        if 'error' in stats:
            print(f"❌ Error getting stats: {stats['error']}")
            return
        
        print("\n📈 Error Statistics (Last 24 hours):")
        for level, data in stats.get('last_24h_by_level', {}).items():
            print(f"  {level}: {data['count']} errors from {data['unique_loggers']} loggers")
        
        print(f"\n📊 Total errors in last 24h: {stats.get('total_errors', 0)}")
        
        print("\n🔍 Recent errors:")
        for error in stats.get('recent_errors', [])[:5]:
            print(f"  [{error['timestamp'].strftime('%H:%M:%S')}] {error['level']}: {error['message'][:60]}...")
        
        # Direct query to verify our test errors
        query = """
        SELECT COUNT(*) 
        FROM system_errors 
        WHERE message LIKE '%test%' 
        AND timestamp >= NOW() - INTERVAL '1 minute'
        """
        with db_ops.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text(query))
            row = result.fetchone()
            test_count = row[0] if row else 0
        
        print(f"\n✅ Found {test_count} test errors in database")
        
        if test_count >= 3:
            print("\n🎉 Error logging to database is working correctly!")
        else:
            print("\n⚠️ Warning: Expected at least 3 test errors, but found only", test_count)
            print("   The async handler may need more time to flush.")
            
    except Exception as e:
        print(f"\n❌ Error checking database: {e}")
        print("   Make sure the database is running and system_errors table exists.")


if __name__ == "__main__":
    test_error_logging()