#!/usr/bin/env python3
"""
Basic test of core functionality without complex imports
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from data.data_utils import create_sample_data
from data.csts_dataset import CSTSDataset
from evaluation.metrics import EvaluationMetrics
from utils.config import get_default_config


def setup_logging():
    """Setup basic logging"""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )


def test_data_and_config():
    """Test data creation and configuration"""
    logger.info("🧪 Testing data creation and configuration...")
    
    # Test configuration
    config = get_default_config()
    assert "dataset" in config
    assert "models" in config
    assert "training" in config
    logger.info("✅ Configuration loaded successfully")
    
    # Test data creation
    data_dir = "./test_basic_data"
    success = create_sample_data(data_dir)
    assert success
    logger.info("✅ Sample data created successfully")
    
    # Test dataset loading
    dataset = CSTSDataset(
        data_path=os.path.join(data_dir, "train.jsonl"),
        tokenizer=None,
        return_labels=True
    )
    assert len(dataset) > 0
    logger.info(f"✅ Dataset loaded successfully - {len(dataset)} examples")
    
    # Test dataset item
    item = dataset[0]
    assert "sentence1" in item
    assert "sentence2" in item
    logger.info(f"✅ Dataset item structure valid")
    
    # Test evaluation metrics
    metrics = EvaluationMetrics()
    predictions = [0.8, 0.6, 0.4, 0.9, 0.3]
    labels = [0.85, 0.65, 0.35, 0.95, 0.25]
    results = metrics.compute_similarity_metrics(predictions, labels)
    
    assert "spearman" in results
    assert "pearson" in results
    logger.info(f"✅ Evaluation metrics computed - Spearman: {results['spearman']:.3f}")
    
    # Clean up
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)
    
    logger.info("✅ All basic tests passed!")


def main():
    """Run basic tests"""
    setup_logging()
    
    logger.info("🚀 Starting basic functionality tests...")
    logger.info("=" * 50)
    
    try:
        test_data_and_config()
        
        logger.info("=" * 50)
        logger.info("🎉 Basic tests completed successfully!")
        logger.info("\n📋 System components verified:")
        logger.info("  ✅ Configuration system")
        logger.info("  ✅ Data creation and loading")
        logger.info("  ✅ Dataset preprocessing")
        logger.info("  ✅ Evaluation metrics")
        
        logger.info("\n🚀 Ready for embedding model distillation!")
        logger.info("\n📚 Next steps:")
        logger.info("  1. Run: python train.py --create_sample")
        logger.info("  2. Customize configs/default_config.yaml")
        logger.info("  3. Use larger models for production training")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())