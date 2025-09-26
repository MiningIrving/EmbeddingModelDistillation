"""
Loss functions for embedding model distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from loguru import logger


class DistillationLoss(nn.Module):
    """
    Combined loss function for embedding model distillation
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        loss_type: str = "mse",
        similarity_loss_weight: float = 1.0,
        embedding_loss_weight: float = 1.0
    ):
        """
        Initialize distillation loss
        
        Args:
            temperature: Temperature for knowledge distillation
            alpha: Weight for distillation loss
            beta: Weight for ground truth loss  
            loss_type: Type of loss ('mse', 'kl_div', 'cosine')
            similarity_loss_weight: Weight for similarity distillation
            embedding_loss_weight: Weight for embedding distillation
        """
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        self.similarity_loss_weight = similarity_loss_weight
        self.embedding_loss_weight = embedding_loss_weight
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        
        logger.info(f"Distillation loss initialized with:")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Alpha (distillation): {alpha}")
        logger.info(f"  Beta (ground truth): {beta}")
        logger.info(f"  Loss type: {loss_type}")
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs  
            labels: Ground truth similarity labels
            
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Embedding distillation loss
        if "embeddings1" in student_outputs and "embeddings1" in teacher_outputs:
            emb_loss1 = self._compute_embedding_loss(
                student_outputs["embeddings1"],
                teacher_outputs["embeddings1"]
            )
            emb_loss2 = self._compute_embedding_loss(
                student_outputs["embeddings2"],
                teacher_outputs["embeddings2"]
            )
            
            embedding_loss = (emb_loss1 + emb_loss2) / 2.0
            losses["embedding_loss"] = embedding_loss
            total_loss += self.embedding_loss_weight * embedding_loss
        
        # 2. Similarity distillation loss
        if "similarities" in student_outputs and "similarities" in teacher_outputs:
            similarity_distill_loss = self._compute_similarity_distillation_loss(
                student_outputs["similarities"],
                teacher_outputs["similarities"]
            )
            losses["similarity_distillation_loss"] = similarity_distill_loss
            total_loss += self.alpha * self.similarity_loss_weight * similarity_distill_loss
        
        # 3. Ground truth similarity loss (if labels available)
        if labels is not None and "similarities" in student_outputs:
            gt_similarity_loss = self._compute_ground_truth_loss(
                student_outputs["similarities"],
                labels
            )
            losses["ground_truth_loss"] = gt_similarity_loss
            total_loss += self.beta * gt_similarity_loss
        
        # 4. Alternative similarity prediction loss (using learned head)
        if "learned_similarities" in student_outputs:
            if labels is not None:
                learned_loss = self.mse_loss(
                    student_outputs["learned_similarities"],
                    labels
                )
                losses["learned_similarity_loss"] = learned_loss
                total_loss += 0.1 * learned_loss  # Small weight
        
        losses["total_loss"] = total_loss
        return losses
    
    def _compute_embedding_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between student and teacher embeddings"""
        
        if self.loss_type == "mse":
            return self.mse_loss(student_embeddings, teacher_embeddings)
        
        elif self.loss_type == "cosine":
            # Use cosine embedding loss with target = 1 (similar embeddings)
            targets = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
            return self.cosine_loss(student_embeddings, teacher_embeddings, targets)
        
        elif self.loss_type == "kl_div":
            # Convert embeddings to probabilities using softmax
            student_probs = F.log_softmax(student_embeddings / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_embeddings / self.temperature, dim=1)
            return self.kl_div_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _compute_similarity_distillation_loss(
        self,
        student_similarities: torch.Tensor,
        teacher_similarities: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss for similarity predictions"""
        
        if self.loss_type == "mse":
            return self.mse_loss(student_similarities, teacher_similarities)
        
        elif self.loss_type == "kl_div":
            # Apply temperature scaling and convert to probabilities
            student_logits = student_similarities / self.temperature
            teacher_logits = teacher_similarities / self.temperature
            
            # Convert to log probabilities (for KL divergence)
            student_log_probs = F.log_softmax(student_logits.unsqueeze(1), dim=1).squeeze(1)
            teacher_probs = F.softmax(teacher_logits.unsqueeze(1), dim=1).squeeze(1)
            
            return F.kl_div(
                student_log_probs.unsqueeze(1),
                teacher_probs.unsqueeze(1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
        
        else:
            return self.mse_loss(student_similarities, teacher_similarities)
    
    def _compute_ground_truth_loss(
        self,
        student_similarities: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss against ground truth similarity labels"""
        return self.mse_loss(student_similarities, labels)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for embedding learning
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: Binary labels (1 for similar, 0 for dissimilar)
            
        Returns:
            Contrastive loss
        """
        # Compute cosine similarity
        similarities = F.cosine_similarity(embeddings1, embeddings2)
        
        # Convert similarities to distances
        distances = 1 - similarities
        
        # Contrastive loss formula
        positive_loss = labels * torch.pow(distances, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0), 2)
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for embedding learning
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor_embeddings: Anchor embeddings
            positive_embeddings: Positive embeddings (similar to anchor)
            negative_embeddings: Negative embeddings (dissimilar to anchor)
            
        Returns:
            Triplet loss
        """
        return self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)