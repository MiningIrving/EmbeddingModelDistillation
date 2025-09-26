"""
Student model implementation for distillation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Any, Optional
from loguru import logger
from .embedding_model import EmbeddingModel


class StudentModel(EmbeddingModel):
    """
    Student model for knowledge distillation
    
    This model will be trained to mimic the teacher model's behavior
    while being more efficient (smaller, faster).
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize student model"""
        super().__init__(*args, **kwargs)
        
        # Student model is trainable
        self.unfreeze_parameters()
        
        # Initialize additional layers for distillation if needed
        self._init_distillation_layers()
        
        logger.info(f"Student model initialized: {self.model_name_or_path}")
        logger.info(f"Student model parameters: {self.get_parameter_count():,}")
        logger.info(f"Trainable parameters: {self.get_trainable_parameter_count():,}")
    
    def _init_distillation_layers(self):
        """Initialize additional layers for distillation training"""
        # Get hidden size from model config
        hidden_size = self.config.hidden_size
        
        # Optional projection layer for dimension matching
        self.projection_layer = None
        
        # Optional temperature scaling for similarity prediction
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Similarity prediction head
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        logger.info("Distillation layers initialized")
    
    def forward(
        self,
        input_ids1: torch.Tensor,
        attention_mask1: torch.Tensor,
        input_ids2: torch.Tensor,
        attention_mask2: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with sentence pairs
        
        Args:
            input_ids1: Input ids for first sentences
            attention_mask1: Attention mask for first sentences
            input_ids2: Input ids for second sentences
            attention_mask2: Attention mask for second sentences
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing embeddings and predictions
        """
        # Encode first sentences
        outputs1 = super().forward(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            **kwargs
        )
        embeddings1 = outputs1["embeddings"]
        
        # Encode second sentences
        outputs2 = super().forward(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            **kwargs
        )
        embeddings2 = outputs2["embeddings"]
        
        # Apply projection if needed
        if self.projection_layer is not None:
            embeddings1 = self.projection_layer(embeddings1)
            embeddings2 = self.projection_layer(embeddings2)
        
        # Compute similarity using cosine similarity
        cosine_similarities = torch.nn.functional.cosine_similarity(
            embeddings1, embeddings2, dim=1
        )
        
        # Alternative similarity using learned head
        combined_embeddings = torch.cat([embeddings1, embeddings2], dim=1)
        learned_similarities = self.similarity_head(combined_embeddings).squeeze(1)
        
        return {
            "embeddings1": embeddings1,
            "embeddings2": embeddings2,
            "cosine_similarities": cosine_similarities,
            "learned_similarities": learned_similarities,
            "temperature": self.temperature
        }
    
    def encode_pairs(
        self,
        sentences1: List[str],
        sentences2: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Encode sentence pairs for evaluation
        
        Args:
            sentences1: First sentences in pairs
            sentences2: Second sentences in pairs
            batch_size: Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing embeddings and similarities
        """
        all_embeddings1 = []
        all_embeddings2 = []
        all_similarities = []
        
        # Process in batches
        for i in range(0, len(sentences1), batch_size):
            batch_sent1 = sentences1[i:i + batch_size]
            batch_sent2 = sentences2[i:i + batch_size]
            
            # Tokenize
            inputs1 = self.tokenizer(
                batch_sent1,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs2 = self.tokenizer(
                batch_sent2,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids1=inputs1["input_ids"],
                    attention_mask1=inputs1["attention_mask"],
                    input_ids2=inputs2["input_ids"],
                    attention_mask2=inputs2["attention_mask"]
                )
            
            all_embeddings1.append(outputs["embeddings1"])
            all_embeddings2.append(outputs["embeddings2"])
            all_similarities.append(outputs["cosine_similarities"])
        
        # Concatenate results
        return {
            "embeddings1": torch.cat(all_embeddings1, dim=0),
            "embeddings2": torch.cat(all_embeddings2, dim=0),
            "similarities": torch.cat(all_similarities, dim=0)
        }
    
    def add_projection_layer(self, target_dim: int):
        """
        Add projection layer to match teacher embedding dimension
        
        Args:
            target_dim: Target embedding dimension
        """
        current_dim = self.config.hidden_size
        
        if current_dim != target_dim:
            self.projection_layer = nn.Linear(current_dim, target_dim)
            self.projection_layer = self.projection_layer.to(self.device)
            logger.info(f"Added projection layer: {current_dim} -> {target_dim}")
        else:
            logger.info("No projection layer needed - dimensions match")
    
    def get_distillation_parameters(self) -> List[nn.Parameter]:
        """Get parameters specific to distillation (excluding base model)"""
        distillation_params = []
        
        if self.projection_layer is not None:
            distillation_params.extend(list(self.projection_layer.parameters()))
        
        distillation_params.extend(list(self.similarity_head.parameters()))
        distillation_params.append(self.temperature)
        
        return distillation_params
    
    def freeze_base_model(self):
        """Freeze base model parameters, keep distillation layers trainable"""
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Keep distillation layers trainable
        for param in self.get_distillation_parameters():
            param.requires_grad = True
        
        logger.info("Base model frozen, distillation layers remain trainable")
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All student model parameters unfrozen")