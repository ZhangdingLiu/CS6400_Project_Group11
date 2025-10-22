"""
Logging Utilities

Functionality: Logging and debugging tools
"""

import logging
import time
from functools import wraps


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    raise NotImplementedError


def timer(func):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to measure

    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start) * 1000:.2f} ms")
        return result
    return wrapper
