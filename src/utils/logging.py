"""
Logging utilities
"""

import os
from loguru import logger
from typing import Dict, Any


def setup_logging(logging_config: Dict[str, Any]):
    """
    Setup logging configuration
    
    Args:
        logging_config: Logging configuration dictionary
    """
    log_level = logging_config.get("log_level", "INFO")
    log_dir = logging_config.get("log_dir", "./logs")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler
    log_file = os.path.join(log_dir, "training.log")
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8"
    )
    
    logger.info("Logging setup completed")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir}")