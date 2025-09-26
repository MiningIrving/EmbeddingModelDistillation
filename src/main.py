#!/usr/bin/env python3
"""
Main entry point for embedding model distillation
"""

import argparse
import os
import sys
from typing import Dict, Any
from loguru import logger

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import download_csts_data, preprocess_csts_data, create_sample_data, create_csts_datasets
from models import TeacherModel, StudentModel
from training import DistillationTrainer
from training.utils import setup_training
from evaluation import ModelEvaluator
from utils import load_config, merge_configs, validate_config
from utils.config import get_default_config, update_config_from_args


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Embedding Model Distillation - 使用开源中文数据集蒸馏现有嵌入模型"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--download_data",
        action="store_true",
        help="Download CSTS dataset"
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample data for testing"
    )
    
    # Model arguments
    parser.add_argument(
        "--teacher_model",
        type=str,
        help="Teacher model name or path"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        help="Student model name or path"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Distillation loss weight"
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Ground truth loss weight"
    )
    
    # Action arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "prepare_data"],
        default="train",
        help="Mode to run the script in"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only run evaluation"
    )
    
    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    return parser.parse_args()


def prepare_data(args, config: Dict[str, Any]) -> bool:
    """
    Prepare dataset for training
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Success status
    """
    data_dir = config["dataset"]["data_dir"]
    
    logger.info("Preparing dataset...")
    
    # Create sample data if requested
    if args.create_sample_data:
        logger.info("Creating sample data...")
        success = create_sample_data(data_dir)
        if not success:
            logger.error("Failed to create sample data")
            return False
        logger.info("Sample data created successfully")
        return True
    
    # Download data if requested
    if args.download_data:
        logger.info("Downloading CSTS dataset...")
        success = download_csts_data(data_dir)
        if not success:
            logger.error("Failed to download dataset")
            return False
        
        # Preprocess downloaded data
        logger.info("Preprocessing dataset...")
        success = preprocess_csts_data(data_dir)
        if not success:
            logger.error("Failed to preprocess dataset")
            return False
        
        logger.info("Dataset prepared successfully")
        return True
    
    # Check if data exists
    train_file = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_file):
        logger.warning(f"Training data not found at {train_file}")
        logger.info("Creating sample data as fallback...")
        return create_sample_data(data_dir)
    
    logger.info("Dataset already prepared")
    return True


def load_models(config: Dict[str, Any]) -> tuple:
    """
    Load teacher and student models
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (teacher_model, student_model)
    """
    logger.info("Loading models...")
    
    # Load teacher model
    teacher_config = config["models"]["teacher"]
    logger.info(f"Loading teacher model: {teacher_config['model_name_or_path']}")
    teacher_model = TeacherModel(
        model_name_or_path=teacher_config["model_name_or_path"],
        max_length=teacher_config.get("max_length", 512),
        device=teacher_config.get("device", "auto")
    )
    
    # Load student model
    student_config = config["models"]["student"]
    logger.info(f"Loading student model: {student_config['model_name_or_path']}")
    student_model = StudentModel(
        model_name_or_path=student_config["model_name_or_path"],
        max_length=student_config.get("max_length", 512),
        device=student_config.get("device", "auto")
    )
    
    # Add projection layer if dimensions don't match
    if teacher_model.config.hidden_size != student_model.config.hidden_size:
        student_model.add_projection_layer(teacher_model.config.hidden_size)
    
    logger.info("Models loaded successfully")
    return teacher_model, student_model


def train_model(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train the distillation model
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Training results
    """
    logger.info("Starting training...")
    
    # Load models
    teacher_model, student_model = load_models(config)
    
    # Load datasets
    datasets = create_csts_datasets(
        data_dir=config["dataset"]["data_dir"],
        tokenizer=student_model.tokenizer,
        max_length=config["dataset"]["max_length"],
        splits=["train", "dev", "test"]
    )
    
    if "train" not in datasets:
        raise ValueError("Training dataset not found")
    
    train_dataset = datasets["train"]
    eval_dataset = datasets.get("dev", None)
    
    logger.info(f"Training dataset: {len(train_dataset)} examples")
    if eval_dataset:
        logger.info(f"Evaluation dataset: {len(eval_dataset)} examples")
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    results = trainer.train()
    
    logger.info("Training completed successfully")
    return results


def evaluate_model(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the trained model
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    logger.info("Starting evaluation...")
    
    # Load models
    teacher_model, student_model = load_models(config)
    
    # Load test dataset
    datasets = create_csts_datasets(
        data_dir=config["dataset"]["data_dir"],
        tokenizer=student_model.tokenizer,
        max_length=config["dataset"]["max_length"],
        splits=["test"]
    )
    
    if "test" not in datasets:
        raise ValueError("Test dataset not found")
    
    test_dataset = datasets["test"]
    logger.info(f"Test dataset: {len(test_dataset)} examples")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    results = evaluator.evaluate_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset=test_dataset,
        batch_size=config["evaluation"]["eval_batch_size"]
    )
    
    logger.info("Evaluation completed successfully")
    return results


def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
        else:
            logger.warning(f"Configuration file not found: {args.config}")
            logger.info("Using default configuration")
            config = get_default_config()
        
        # Update config with command line arguments
        config = update_config_from_args(config, vars(args))
        
        # Validate configuration
        validate_config(config)
        
        # Setup training environment
        config = setup_training(config)
        
        # Enable wandb if requested
        if args.wandb:
            config["logging"]["wandb"]["enabled"] = True
        
        # Run based on mode
        if args.mode == "prepare_data" or args.download_data or args.create_sample_data:
            success = prepare_data(args, config)
            if not success:
                sys.exit(1)
        
        elif args.mode == "evaluate" or args.evaluate_only:
            results = evaluate_model(args, config)
            logger.info("Evaluation Results:")
            logger.info(f"Teacher Spearman: {results['teacher_results']['similarity_metrics']['spearman']:.4f}")
            logger.info(f"Student Spearman: {results['student_results']['similarity_metrics']['spearman']:.4f}")
        
        elif args.mode == "train":
            # Prepare data if needed
            prepare_data(args, config)
            
            # Train model
            results = train_model(args, config)
            logger.info(f"Training completed. Best metric: {results['best_metric']:.4f}")
            
            # Evaluate if test data available
            if os.path.exists(os.path.join(config["dataset"]["data_dir"], "test.jsonl")):
                logger.info("Running final evaluation...")
                eval_results = evaluate_model(args, config)
                logger.info("Final Evaluation Results:")
                logger.info(f"Teacher Spearman: {eval_results['teacher_results']['similarity_metrics']['spearman']:.4f}")
                logger.info(f"Student Spearman: {eval_results['student_results']['similarity_metrics']['spearman']:.4f}")
        
        logger.info("Script completed successfully")
        
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()