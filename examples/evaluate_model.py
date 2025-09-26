#!/usr/bin/env python3
"""
Example script for evaluating embedding models
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import main


def example_evaluation():
    """Run example evaluation"""
    
    # Simulate command line arguments
    sys.argv = [
        "evaluate_model.py",
        "--mode", "evaluate",
        "--config", "configs/default_config.yaml",
        "--create_sample_data",  # Use sample data for this example
        "--output_dir", "./example_output",
        "--log_level", "INFO"
    ]
    
    print("Running embedding model evaluation example...")
    print("=" * 60)
    
    # Run main function
    main()
    
    print("=" * 60)
    print("Evaluation completed!")


if __name__ == "__main__":
    example_evaluation()