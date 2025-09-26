#!/usr/bin/env python3
"""
Example script for training embedding model distillation
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import main
import argparse


def example_training():
    """Run example training with sample data"""
    
    # Simulate command line arguments
    sys.argv = [
        "train_distillation.py",
        "--mode", "train",
        "--config", "configs/default_config.yaml",
        "--create_sample_data",  # Use sample data for this example
        "--num_epochs", "2",     # Short training for example
        "--batch_size", "8",     # Small batch size
        "--output_dir", "./example_output",
        "--log_level", "INFO"
    ]
    
    print("Running embedding model distillation example...")
    print("=" * 60)
    
    # Run main function
    main()
    
    print("=" * 60)
    print("Example completed!")
    print("Check ./example_output for results")


if __name__ == "__main__":
    example_training()