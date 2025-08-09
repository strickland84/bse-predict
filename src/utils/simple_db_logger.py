"""Simple database error logger without circular dependencies."""
import logging
import os
import psycopg2
from datetime import datetime
import traceback as tb
import socket
import threading


class SimpleDBErrorHandler(logging.Handler):
    """Dead simple database error handler - no fancy imports."""
    
    def __init__(self, db_url=None, level=logging.ERROR):
        super().__init__(level)
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        
    def emit(self, record):
        """Write error to database."""
        # Skip if no DB URL or if we're in a DB operation
        if not self.db_url or getattr(record, 'skip_db_log', False):
            return
            
        # Skip if below error level
        if record.levelno < logging.ERROR:
            return
            
        try:
            # Extract exception info if present
            exc_text = None
            tb_text = None
            if record.exc_info:
                exc_text = str(record.exc_info[1])
                tb_text = ''.join(tb.format_exception(*record.exc_info))
            
            # Simple direct insert
            conn = psycopg2.connect(self.db_url)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO system_errors 
                        (timestamp, logger_name, level, message, exception, traceback,
                         module, function, line_number, thread_name, process_id, hostname)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        datetime.utcnow(),
                        record.name,
                        record.levelname,
                        self.format(record),
                        exc_text,
                        tb_text,
                        record.module,
                        record.funcName,
                        record.lineno,
                        record.threadName,
                        self.pid,
                        self.hostname
                    ))
                    conn.commit()
            finally:
                conn.close()
        except:
            # Silently fail - we don't want error logging to break the app
            pass


def add_db_error_handler(logger=None, db_url=None):
    """Add database error handler to a logger."""
    if logger is None:
        logger = logging.getLogger()
    
    # Remove any existing DB handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, SimpleDBErrorHandler):
            logger.removeHandler(handler)
    
    # Add new handler
    handler = SimpleDBErrorHandler(db_url)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    
    return handler