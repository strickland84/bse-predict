"""Logging configuration for BSE Predict."""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from .config import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_size: str = "10MB",
    backup_count: int = 5,
    use_db_handler: bool = True,
) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_size: Maximum log file size (e.g., "10MB")
        backup_count: Number of backup files to keep
        use_db_handler: Whether to enable database error logging
    """
    if log_level is None:
        log_level = config.log_level
    
    if log_file is None:
        log_file = config.get_nested("logging.file", "logs/app.log")
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse max_size
    max_bytes = _parse_size(max_size)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        config.get_nested(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Database handler for errors (if enabled and not in test environment)
    if use_db_handler and os.getenv('TESTING') != 'true':
        try:
            from .db_logger import AsyncDatabaseLogHandler
            db_handler = AsyncDatabaseLogHandler(level=logging.ERROR)
            db_handler.setFormatter(formatter)
            root_logger.addHandler(db_handler)
            logger = logging.getLogger(__name__)
            logger.info("Database error logging enabled")
        except Exception as e:
            # If database handler fails, continue without it
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not enable database error logging: {e}")
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes."""
    size_str = size_str.upper()
    
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            number = size_str[:-len(suffix)]
            try:
                return int(float(number) * multiplier)
            except ValueError:
                pass
    
    # Default to 10MB if parsing fails
    return 10 * 1024 * 1024


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Setup logging on import - but disable DB handler to avoid circular imports
setup_logging(use_db_handler=False)
