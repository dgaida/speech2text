"""Logging utilities for the speech2text module."""

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose (bool): Whether to enable verbose output.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)
