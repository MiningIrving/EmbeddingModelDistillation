#!/usr/bin/env python3
"""
Demo script for embedding model distillation
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from data.data_utils import create_sample_data, load_dataset_stats
from data.csts_dataset import CSTSDataset
from evaluation.metrics import EvaluationMetrics
from utils.config import get_default_config, save_config


def setup_logging():
    """Setup demo logging"""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<cyan>{time:HH:mm:ss}</cyan> | <level>{message}</level>"
    )


def main():
    """Demo the embedding model distillation system"""
    setup_logging()
    
    logger.info("🎯 Embedding Model Distillation Demo")
    logger.info("=" * 60)
    
    # Configuration demo
    logger.info("📋 Loading configuration...")
    config = get_default_config()
    logger.info(f"  Teacher model: {config['models']['teacher']['model_name_or_path']}")
    logger.info(f"  Student model: {config['models']['student']['model_name_or_path']}")
    logger.info(f"  Training epochs: {config['training']['num_epochs']}")
    logger.info(f"  Distillation temperature: {config['distillation']['temperature']}")
    
    # Data demo
    logger.info("\n📊 Creating sample Chinese STS dataset...")
    data_dir = "./demo_data"
    success = create_sample_data(data_dir)
    
    if success:
        stats = load_dataset_stats(data_dir)
        for split, info in stats.items():
            logger.info(f"  {split}: {info['num_examples']} examples (score: {info['score_min']:.1f}-{info['score_max']:.1f})")
    
    # Dataset demo
    logger.info("\n📚 Loading dataset...")
    dataset = CSTSDataset(
        data_path=os.path.join(data_dir, "train.jsonl"),
        return_labels=True
    )
    
    logger.info(f"  Dataset size: {len(dataset)} examples")
    
    # Show sample data
    logger.info("\n🔍 Sample data:")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        logger.info(f"  {i+1}. '{item['sentence1']}' | '{item['sentence2']}' | score: {item.get('raw_score', 'N/A')}")
    
    # Evaluation demo
    logger.info("\n📈 Evaluation metrics demo...")
    # Simulate some predictions vs labels
    predictions = [0.85, 0.62, 0.41, 0.88, 0.29, 0.74, 0.55, 0.91, 0.33, 0.78]
    labels = [0.90, 0.65, 0.35, 0.85, 0.25, 0.70, 0.60, 0.95, 0.30, 0.75]
    
    metrics = EvaluationMetrics()
    results = metrics.compute_similarity_metrics(predictions, labels)
    
    logger.info("  Correlation metrics:")
    logger.info(f"    Spearman correlation: {results['spearman']:.4f}")
    logger.info(f"    Pearson correlation: {results['pearson']:.4f}")
    logger.info("  Regression metrics:")
    logger.info(f"    MSE: {results['mse']:.4f}")
    logger.info(f"    MAE: {results['mae']:.4f}")
    logger.info("  Classification metrics:")
    logger.info(f"    Accuracy: {results['accuracy']:.4f}")
    logger.info(f"    F1 Score: {results['f1']:.4f}")
    
    # Save demo config
    logger.info("\n💾 Saving demo configuration...")
    demo_config_path = "./demo_config.yaml"
    config["dataset"]["data_dir"] = data_dir
    config["training"]["num_epochs"] = 2  # Quick demo training
    config["training"]["batch_size"] = 8  # Small batch for demo
    save_config(config, demo_config_path)
    
    # Usage instructions
    logger.info("\n" + "=" * 60)
    logger.info("🚀 System Ready! Next Steps:")
    logger.info("\n1️⃣ Quick Start (with sample data):")
    logger.info("   python train.py --create_sample --data_dir ./demo_data")
    
    logger.info("\n2️⃣ Download Real CSTS Dataset:")
    logger.info("   python train.py --download_data --data_dir ./data/csts")
    
    logger.info("\n3️⃣ Customize Configuration:")
    logger.info("   # Edit configs/default_config.yaml")
    logger.info("   # Adjust models, hyperparameters, etc.")
    
    logger.info("\n4️⃣ Training Pipeline:")
    logger.info("   # The full training would use:")
    logger.info("   # - Teacher model: qzhou-embedding (based on Qwen2.5-7B)")
    logger.info("   # - Student model: qwen3-embedding-4b (based on Qwen2.5-3B)")
    logger.info("   # - Knowledge distillation with temperature scaling")
    logger.info("   # - Multiple loss functions (MSE, KL-div, cosine)")
    logger.info("   # - Comprehensive evaluation metrics")
    
    logger.info("\n5️⃣ Model Architecture:")
    logger.info("   📊 Dataset: CSTS (Chinese Semantic Textual Similarity)")
    logger.info("   🎯 Task: Embedding similarity prediction")
    logger.info("   🧠 Method: Knowledge distillation")
    logger.info("   📈 Metrics: Spearman, Pearson, Accuracy, F1")
    
    logger.info("\n" + "=" * 60)
    logger.info("✨ Demo completed successfully!")
    logger.info("🔥 Ready for Chinese embedding model distillation!")
    
    # Clean up demo data
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()