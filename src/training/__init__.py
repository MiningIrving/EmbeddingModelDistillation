"""
Training components for embedding model distillation
"""

from .distillation_trainer import DistillationTrainer
from .losses import DistillationLoss
from .utils import setup_training, save_checkpoint, load_checkpoint

__all__ = ["DistillationTrainer", "DistillationLoss", "setup_training", "save_checkpoint", "load_checkpoint"]