"""
Teacher model implementation for distillation
"""

import torch
from typing import Dict, List, Union, Any
from loguru import logger
from .embedding_model import EmbeddingModel


class TeacherModel(EmbeddingModel):
    """
    Teacher model for knowledge distillation
    
    This model will be frozen during training and used to generate
    target embeddings for the student model to learn from.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize teacher model"""
        super().__init__(*args, **kwargs)
        
        # Freeze teacher model parameters
        self.freeze_parameters()
        
        # Set to evaluation mode
        self.eval()
        
        logger.info(f"Teacher model initialized and frozen: {self.model_name_or_path}")
        logger.info(f"Teacher model parameters: {self.get_parameter_count():,}")
    
    def generate_targets(
        self,
        sentences1: List[str],
        sentences2: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate target embeddings for student training
        
        Args:
            sentences1: First sentences in pairs
            sentences2: Second sentences in pairs
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Dictionary containing target embeddings and similarities
        """
        with torch.no_grad():
            # Encode both sets of sentences
            embeddings1 = self.encode(
                sentences1,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings
            )
            embeddings2 = self.encode(
                sentences2,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings
            )
            
            # Compute similarities
            similarities = torch.nn.functional.cosine_similarity(
                embeddings1, embeddings2, dim=1
            )
            
            return {
                "embeddings1": embeddings1,
                "embeddings2": embeddings2,
                "similarities": similarities
            }
    
    def forward(self, *args, **kwargs):
        """Forward pass - always in no_grad mode for teacher"""
        with torch.no_grad():
            return super().forward(*args, **kwargs)
    
    def train(self, mode: bool = True):
        """Override train mode - teacher is always in eval mode"""
        return super().train(False)  # Always keep in eval mode
    
    def freeze_parameters(self):
        """Freeze all parameters (override to add logging)"""
        super().freeze_parameters()
        logger.info("Teacher model parameters frozen")
    
    def unfreeze_parameters(self):
        """Prevent unfreezing teacher parameters"""
        logger.warning("Cannot unfreeze teacher model parameters")
        pass