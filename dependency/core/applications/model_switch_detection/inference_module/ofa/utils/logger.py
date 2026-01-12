# utils/logger.py
import logging
import sys
from datetime import datetime
import os

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a logger instance.

    Args:
        name: Logger name.
        log_dir: Directory to store log files.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # If the logger already has handlers, do not add new handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create log directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Add file handler
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{current_time}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create a default logger
default_logger = setup_logger('default')