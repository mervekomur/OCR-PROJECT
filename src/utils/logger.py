"""
Logging utility for OCR project.
Centralized logging configuration.
"""

import logging
import sys
from typing import Optional


# Default format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '[%(levelname)s] %(message)s'


def setup_logging(
    level: int = logging.INFO,
    format_string: str = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
        log_file: Optional file path for logging
    """
    format_str = format_string or SIMPLE_FORMAT

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set default handler if not configured
    if not logger.handlers and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(SIMPLE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
