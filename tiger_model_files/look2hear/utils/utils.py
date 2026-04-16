import logging
import sys

def get_logger(name):
    """
    Gets a logger instance that prints to the console.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Create a handler that writes to standard output
        handler = logging.StreamHandler(sys.stdout)
        # Create a formatter to define the log message format
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Add the formatter to the handler
        handler.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(handler)
    return logger

def print_only(message):
    """
    A simple utility to print messages.
    Can be expanded later to support distributed training (e.g., print only on rank 0).
    """
    print(message)