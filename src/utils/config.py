"""
Configuration utilities
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from loguru import logger
import copy


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


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, "w", encoding="utf-8") as f:
        if file_ext in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif file_ext == ".json":
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
    
    logger.info(f"Configuration saved to: {config_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries recursively
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = copy.deepcopy(base_config)
    
    def _merge_dict(base_dict: Dict, override_dict: Dict):
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                _merge_dict(base_dict[key], value)
            else:
                base_dict[key] = copy.deepcopy(value)
    
    _merge_dict(merged, override_config)
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises exception if invalid
    """
    required_sections = ["dataset", "models", "training"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate dataset config
    dataset_config = config["dataset"]
    required_dataset_keys = ["name", "data_dir", "batch_size"]
    for key in required_dataset_keys:
        if key not in dataset_config:
            raise ValueError(f"Missing required dataset configuration: {key}")
    
    # Validate models config
    models_config = config["models"]
    if "teacher" not in models_config or "student" not in models_config:
        raise ValueError("Both teacher and student model configurations are required")
    
    for model_type in ["teacher", "student"]:
        model_config = models_config[model_type]
        if "model_name_or_path" not in model_config:
            raise ValueError(f"Missing model_name_or_path for {model_type} model")
    
    # Validate training config
    training_config = config["training"]
    required_training_keys = ["output_dir", "num_epochs", "learning_rate"]
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training configuration: {key}")
    
    logger.info("Configuration validation passed")
    return True


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        "dataset": {
            "name": "csts",
            "data_dir": "./data/csts",
            "train_file": "train.jsonl",
            "dev_file": "dev.jsonl",
            "test_file": "test.jsonl",
            "max_length": 512,
            "batch_size": 32,
            "num_workers": 4
        },
        "models": {
            "teacher": {
                "name": "qzhou-embedding",
                "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                "max_length": 512,
                "device": "auto"
            },
            "student": {
                "name": "qwen3-embedding-4b",
                "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
                "max_length": 512,
                "device": "auto"
            }
        },
        "training": {
            "output_dir": "./output",
            "num_epochs": 5,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 100,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_spearman",
            "greater_is_better": True,
            "dataloader_drop_last": False,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,
            "fp16": True,
            "seed": 42
        },
        "distillation": {
            "temperature": 4.0,
            "alpha": 0.7,
            "beta": 0.3,
            "loss_type": "mse"
        },
        "evaluation": {
            "metrics": ["spearman", "pearson", "accuracy"],
            "eval_batch_size": 64
        },
        "logging": {
            "log_level": "INFO",
            "log_dir": "./logs",
            "wandb": {
                "enabled": False,
                "project": "embedding-distillation",
                "entity": None,
                "name": None
            }
        }
    }


def update_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with command-line arguments
    
    Args:
        config: Base configuration
        args: Command-line arguments
        
    Returns:
        Updated configuration
    """
    # Map command-line arguments to config paths
    arg_mapping = {
        "output_dir": ["training", "output_dir"],
        "learning_rate": ["training", "learning_rate"],
        "batch_size": ["dataset", "batch_size"],
        "num_epochs": ["training", "num_epochs"],
        "teacher_model": ["models", "teacher", "model_name_or_path"],
        "student_model": ["models", "student", "model_name_or_path"],
        "data_dir": ["dataset", "data_dir"],
        "temperature": ["distillation", "temperature"],
        "alpha": ["distillation", "alpha"],
        "beta": ["distillation", "beta"]
    }
    
    updated_config = copy.deepcopy(config)
    
    for arg_name, config_path in arg_mapping.items():
        if arg_name in args and args[arg_name] is not None:
            # Navigate to the nested dictionary location
            current_dict = updated_config
            for key in config_path[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]
            
            # Set the value
            current_dict[config_path[-1]] = args[arg_name]
            logger.info(f"Updated config {'.'.join(config_path)}: {args[arg_name]}")
    
    return updated_config