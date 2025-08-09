"""Database logging handler for storing errors in PostgreSQL."""
import logging
import traceback
import socket
import os
import threading
from datetime import datetime
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor


class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that stores error logs in database."""
    
    def __init__(self, level=logging.ERROR):
        """Initialize the database log handler.
        
        Args:
            level: Minimum log level to capture (default: ERROR)
        """
        super().__init__(level)
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        
    def emit(self, record: logging.LogRecord):
        """Store log record in database.
        
        Args:
            record: LogRecord to store
        """
        # Skip if we're already in a database operation to prevent recursion
        if hasattr(threading.current_thread(), '_in_db_log_handler'):
            return
            
        try:
            # Only log ERROR and CRITICAL to database
            if record.levelno < logging.ERROR:
                return
            
            # Mark that we're in the handler to prevent recursion
            threading.current_thread()._in_db_log_handler = True
                
            # Extract exception info if available
            exception_text = None
            traceback_text = None
            
            if record.exc_info:
                exception_text = str(record.exc_info[1])
                traceback_text = ''.join(traceback.format_exception(*record.exc_info))
            
            # Get database URL from environment only
            import os
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                # Cannot log to database without DATABASE_URL env var
                return
            
            # Direct database connection - no SQLAlchemy, no logging
            conn = psycopg2.connect(db_url)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO system_errors (
                            timestamp, logger_name, level, message,
                            exception, traceback, module, function,
                            line_number, thread_name, process_id, hostname
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        datetime.utcnow(),
                        record.name,
                        record.levelname,
                        self.format(record),
                        exception_text,
                        traceback_text,
                        record.module,
                        record.funcName,
                        record.lineno,
                        record.threadName,
                        self.process_id,
                        self.hostname
                    ))
                    conn.commit()
            finally:
                conn.close()
            
        except Exception:
            # If database logging fails, just ignore it
            pass
        finally:
            # Clear the recursion flag
            if hasattr(threading.current_thread(), '_in_db_log_handler'):
                delattr(threading.current_thread(), '_in_db_log_handler')


class AsyncDatabaseLogHandler(DatabaseLogHandler):
    """Asynchronous database log handler to avoid blocking on database writes."""
    
    def __init__(self, level=logging.ERROR, max_queue_size=1000):
        """Initialize async database log handler.
        
        Args:
            level: Minimum log level to capture
            max_queue_size: Maximum number of records to queue
        """
        super().__init__(level)
        self.queue = []
        self.max_queue_size = max_queue_size
        self.lock = threading.Lock()
        self.timer = None
        self.flush_interval = 5.0  # Flush every 5 seconds
        
    def emit(self, record: logging.LogRecord):
        """Queue log record for later database storage."""
        try:
            with self.lock:
                if len(self.queue) < self.max_queue_size:
                    self.queue.append(record)
                    
                    # Start flush timer if not already running
                    if self.timer is None or not self.timer.is_alive():
                        self.timer = threading.Timer(self.flush_interval, self.flush)
                        self.timer.daemon = True
                        self.timer.start()
                        
                    # Flush immediately if queue is getting full
                    if len(self.queue) >= self.max_queue_size * 0.8:
                        self.flush()
                        
        except Exception:
            self.handleError(record)
            
    def flush(self):
        """Flush queued records to database."""
        with self.lock:
            if not self.queue:
                return
                
            records_to_flush = self.queue[:]
            self.queue = []
            
        # Process records in a separate thread to avoid blocking
        def process_records():
            for record in records_to_flush:
                try:
                    super(AsyncDatabaseLogHandler, self).emit(record)
                except Exception:
                    # If one record fails, continue with others
                    pass
        
        flush_thread = threading.Thread(target=process_records)
        flush_thread.daemon = True
        flush_thread.start()
                
    def close(self):
        """Flush and close the handler."""
        self.flush()
        super().close()


def get_db_logger_stats() -> dict:
    """Get statistics about logged errors from database.
    
    Returns:
        Dictionary with error statistics
    """
    try:
        # Get error counts by level for last 24 hours
        query = """
        SELECT 
            level,
            COUNT(*) as count,
            COUNT(DISTINCT logger_name) as unique_loggers,
            COUNT(DISTINCT module) as unique_modules
        FROM system_errors
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        GROUP BY level
        ORDER BY 
            CASE level
                WHEN 'CRITICAL' THEN 1
                WHEN 'ERROR' THEN 2
                WHEN 'WARNING' THEN 3
                ELSE 4
            END
        """
        
        from src.database.operations import db_ops
        with db_ops.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text(query))
            rows = result.fetchall()
        
        stats = {
            'last_24h_by_level': {},
            'total_errors': 0
        }
        
        for row in rows:
            level, count, unique_loggers, unique_modules = row
            stats['last_24h_by_level'][level] = {
                'count': count,
                'unique_loggers': unique_loggers,
                'unique_modules': unique_modules
            }
            stats['total_errors'] += count
            
        # Get most recent errors
        query = """
        SELECT timestamp, logger_name, level, message
        FROM system_errors
        ORDER BY timestamp DESC
        LIMIT 10
        """
        
        from src.database.operations import db_ops
        with db_ops.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text(query))
            rows = result.fetchall()
        
        stats['recent_errors'] = [
            {
                'timestamp': row[0],
                'logger': row[1],
                'level': row[2],
                'message': row[3][:200]  # Truncate long messages
            }
            for row in rows
        ]
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}