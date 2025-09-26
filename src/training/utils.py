"""
Training utilities for embedding model distillation
"""

import os
import json
import torch
import random
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger
import yaml


def setup_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup training environment
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with runtime settings
    """
    # Set random seeds for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Setup logging
    setup_logging(config.get("logging", {}))
    
    # Setup device
    device = setup_device(config.get("device", "auto"))
    config["device"] = device
    
    # Create output directories
    output_dir = config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info("Training environment setup completed")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {seed}")
    
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def setup_device(device: str = "auto") -> str:
    """Setup compute device"""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {gpu_count} GPU(s)")
            
            # Log GPU information
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    
    return device


def setup_logging(logging_config: Dict[str, Any]):
    """Setup logging configuration"""
    log_level = logging_config.get("log_level", "INFO")
    log_dir = logging_config.get("log_dir", "./logs")
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure loguru
    logger.remove()  # Remove default handler
    
    # Add console handler
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
        retention="10 days"
    )
    
    logger.info("Logging setup completed")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False
) -> str:
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Current step
        loss: Current loss
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)
    
    # Save as best if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"New best checkpoint saved: {best_path}")
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Checkpoint loaded - Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
    
    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "loss": checkpoint["loss"],
        "metrics": checkpoint.get("metrics", {})
    }


def cleanup_checkpoints(checkpoint_dir: str, keep_n: int = 3):
    """
    Clean up old checkpoints, keeping only the most recent ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_n: Number of checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Find all checkpoint files
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
            # Extract epoch and step from filename
            try:
                parts = filename.replace("checkpoint_epoch_", "").replace(".pt", "").split("_step_")
                epoch = int(parts[0])
                step = int(parts[1])
                checkpoint_files.append((epoch, step, filename))
            except:
                continue
    
    # Sort by epoch and step (newest first)
    checkpoint_files.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Remove old checkpoints
    for epoch, step, filename in checkpoint_files[keep_n:]:
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            os.remove(filepath)
            logger.debug(f"Removed old checkpoint: {filename}")
        except:
            pass


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model size and parameter count
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory footprint (approximate)
    memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "memory_mb": memory_mb,
        "memory_gb": memory_mb / 1024
    }


def log_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """Log model information"""
    size_info = calculate_model_size(model)
    
    logger.info(f"{model_name} Information:")
    logger.info(f"  Total parameters: {size_info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {size_info['trainable_parameters']:,}")
    logger.info(f"  Non-trainable parameters: {size_info['non_trainable_parameters']:,}")
    logger.info(f"  Estimated memory: {size_info['memory_gb']:.2f} GB")


def save_training_config(config: Dict[str, Any], output_dir: str):
    """Save training configuration to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training configuration saved: {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, "r", encoding="utf-8") as f:
        if file_ext in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif file_ext == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
    
    logger.info(f"Configuration loaded from: {config_path}")
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    import copy
    
    merged = copy.deepcopy(base_config)
    
    def _merge_dict(base_dict: Dict, override_dict: Dict):
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                _merge_dict(base_dict[key], value)
            else:
                base_dict[key] = copy.deepcopy(value)
    
    _merge_dict(merged, override_config)
    return merged