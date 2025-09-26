"""
Device utilities
"""

import torch
from loguru import logger


def setup_device(device: str = "auto") -> str:
    """
    Setup compute device
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", etc.)
        
    Returns:
        Device string
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {gpu_count} GPU(s)")
            
            # Log GPU information
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_properties = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_properties.total_memory / 1e9
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    else:
        logger.info(f"Using specified device: {device}")
    
    return device


def get_device_info() -> dict:
    """
    Get detailed device information
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_reserved": torch.cuda.memory_reserved(i),
            }
            info["devices"].append(device_info)
    
    return info