"""
Structured logging configuration for the Multimodal RAG system.

Provides consistent logging setup with timestamps, levels, and
contextual information.
"""

import logging
import sys
from typing import Optional

from config.settings import settings


def setup_logger(
    name: str,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses settings.api.log_level.

    Returns:
        Configured logger instance.
    """
    log_level = level or settings.api.log_level
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with formatted output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Formatter with timestamp and level
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


def log_function_call(logger: logging.Logger, func_name: str, **kwargs) -> None:
    """
    Log a function call with parameters.

    Args:
        logger: Logger instance.
        func_name: Function name.
        **kwargs: Parameters to log.
    """
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_function_result(
    logger: logging.Logger, func_name: str, duration: float, result_summary: str
) -> None:
    """
    Log function completion with timing.

    Args:
        logger: Logger instance.
        func_name: Function name.
        duration: Execution duration in seconds.
        result_summary: Summary of the result.
    """
    logger.debug(f"{func_name} completed in {duration:.2f}s - {result_summary}")
