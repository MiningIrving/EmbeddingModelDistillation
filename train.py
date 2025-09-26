#!/usr/bin/env python3
"""
Simple training script for embedding model distillation
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import required modules
from data.data_utils import create_sample_data, download_csts_data, preprocess_csts_data
from data.csts_dataset import create_csts_datasets
from utils.config import load_config, get_default_config
from loguru import logger


def setup_logging():
    """Setup basic logging"""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Embedding Model Distillation Training")
    parser.add_argument("--data_dir", default="./data/csts", help="Data directory")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Config file")
    parser.add_argument("--create_sample", action="store_true", help="Create sample data")
    parser.add_argument("--download_data", action="store_true", help="Download CSTS data")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    setup_logging()
    logger.info("🚀 Starting Embedding Model Distillation")
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create config
    if os.path.exists(args.config):
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        logger.info("Using default configuration")
        config = get_default_config()
    
    # Update config with args
    config["dataset"]["data_dir"] = args.data_dir
    config["training"]["output_dir"] = args.output_dir
    
    # Prepare data
    logger.info("📊 Preparing dataset...")
    
    if args.create_sample:
        logger.info("Creating sample data for testing...")
        success = create_sample_data(args.data_dir)
        if success:
            logger.info("✅ Sample data created successfully")
        else:
            logger.error("❌ Failed to create sample data")
            return 1
    
    elif args.download_data:
        logger.info("Downloading CSTS dataset...")
        success = download_csts_data(args.data_dir)
        if success:
            success = preprocess_csts_data(args.data_dir)
            if success:
                logger.info("✅ Dataset downloaded and preprocessed successfully")
            else:
                logger.error("❌ Failed to preprocess dataset")
                return 1
        else:
            logger.error("❌ Failed to download dataset")
            return 1
    
    else:
        # Check if data exists
        train_file = os.path.join(args.data_dir, "train.jsonl")
        if not os.path.exists(train_file):
            logger.warning("Training data not found, creating sample data...")
            create_sample_data(args.data_dir)
    
    # Load datasets to verify
    try:
        logger.info("📚 Loading datasets...")
        datasets = create_csts_datasets(
            data_dir=args.data_dir,
            tokenizer=None,  # Will be loaded with models
            splits=["train", "dev", "test"]
        )
        
        for split, dataset in datasets.items():
            logger.info(f"  {split}: {len(dataset)} examples")
        
        logger.info("✅ Datasets loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading datasets: {e}")
        return 1
    
    logger.info("🎯 Setup completed successfully!")
    logger.info(f"📁 Data directory: {args.data_dir}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"📄 Config file: {args.config}")
    
    logger.info("\n🔥 To run full training, install the package and use:")
    logger.info("   python -m src.main --mode train --create_sample_data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())