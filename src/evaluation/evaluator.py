"""
Model evaluator for embedding models
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from loguru import logger
from tqdm import tqdm

from ..models import EmbeddingModel, StudentModel, TeacherModel
from ..data import CSTSDataset, CSTSDataLoader
from .metrics import EvaluationMetrics


class ModelEvaluator:
    """
    Evaluator for embedding models
    """
    
    def __init__(self, metrics: Optional[EvaluationMetrics] = None):
        """
        Initialize model evaluator
        
        Args:
            metrics: EvaluationMetrics instance
        """
        self.metrics = metrics or EvaluationMetrics()
    
    def evaluate_model(
        self,
        model: EmbeddingModel,
        dataset: CSTSDataset,
        batch_size: int = 64,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset
        
        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model on {len(dataset)} examples")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create data loader
        dataloader = CSTSDataLoader.create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for evaluation to avoid issues
            drop_last=False
        )
        
        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        all_embeddings1 = []
        all_embeddings2 = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Move to device if specified
                if device:
                    batch = self._move_batch_to_device(batch, device)
                
                if isinstance(model, StudentModel):
                    # Use student-specific evaluation
                    outputs = model.encode_pairs(
                        batch["sentence1"],
                        batch["sentence2"],
                        batch_size=len(batch["sentence1"])
                    )
                    predictions = outputs["similarities"].cpu().numpy()
                    embeddings1 = outputs["embeddings1"].cpu().numpy()
                    embeddings2 = outputs["embeddings2"].cpu().numpy()
                    
                else:
                    # Use general embedding model evaluation
                    embeddings1 = model.encode(batch["sentence1"], return_tensors=False)
                    embeddings2 = model.encode(batch["sentence2"], return_tensors=False)
                    
                    # Compute similarities
                    predictions = np.sum(embeddings1 * embeddings2, axis=1) / (
                        np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
                    )
                
                all_predictions.extend(predictions)
                all_embeddings1.extend(embeddings1)
                all_embeddings2.extend(embeddings2)
                
                # Collect labels if available
                if "labels" in batch:
                    labels = batch["labels"].cpu().numpy() if isinstance(batch["labels"], torch.Tensor) else batch["labels"]
                    all_labels.extend(labels)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_embeddings1 = np.array(all_embeddings1)
        all_embeddings2 = np.array(all_embeddings2)
        
        # Compute evaluation metrics
        results = {}
        
        # Similarity metrics
        if all_labels:
            all_labels = np.array(all_labels)
            similarity_metrics = self.metrics.compute_similarity_metrics(
                all_predictions, all_labels
            )
            results["similarity_metrics"] = similarity_metrics
            
            # Ranking metrics
            ranking_metrics = self.metrics.compute_ranking_metrics(
                all_predictions, all_labels
            )
            results["ranking_metrics"] = ranking_metrics
        
        # Embedding quality metrics
        embedding_metrics1 = self.metrics.compute_embedding_quality_metrics(all_embeddings1)
        embedding_metrics2 = self.metrics.compute_embedding_quality_metrics(all_embeddings2)
        
        results["embedding_metrics"] = {
            "sentence1_embeddings": embedding_metrics1,
            "sentence2_embeddings": embedding_metrics2
        }
        
        # Summary statistics
        results["summary"] = {
            "num_examples": len(all_predictions),
            "prediction_stats": {
                "mean": np.mean(all_predictions),
                "std": np.std(all_predictions),
                "min": np.min(all_predictions),
                "max": np.max(all_predictions)
            }
        }
        
        if all_labels:
            results["summary"]["label_stats"] = {
                "mean": np.mean(all_labels),
                "std": np.std(all_labels),
                "min": np.min(all_labels),
                "max": np.max(all_labels)
            }
        
        logger.info("Evaluation completed")
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, EmbeddingModel],
        dataset: CSTSDataset,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset
        
        Args:
            models: Dictionary of model name -> model
            dataset: Dataset for evaluation
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            results = self.evaluate_model(model, dataset, batch_size)
            comparison_results[model_name] = results
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        comparison_results["comparison_summary"] = summary
        
        return comparison_results
    
    def evaluate_distillation(
        self,
        teacher_model: TeacherModel,
        student_model: StudentModel,
        dataset: CSTSDataset,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Evaluate teacher-student distillation performance
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            dataset: Evaluation dataset
            batch_size: Batch size
            
        Returns:
            Distillation evaluation results
        """
        logger.info("Evaluating distillation performance")
        
        # Evaluate both models
        teacher_results = self.evaluate_model(teacher_model, dataset, batch_size)
        student_results = self.evaluate_model(student_model, dataset, batch_size)
        
        # Compute distillation-specific metrics
        distillation_metrics = self._compute_distillation_metrics(
            teacher_model, student_model, dataset, batch_size
        )
        
        return {
            "teacher_results": teacher_results,
            "student_results": student_results,
            "distillation_metrics": distillation_metrics
        }
    
    def _compute_distillation_metrics(
        self,
        teacher_model: TeacherModel,
        student_model: StudentModel,
        dataset: CSTSDataset,
        batch_size: int
    ) -> Dict[str, float]:
        """Compute distillation-specific metrics"""
        # Create data loader
        dataloader = CSTSDataLoader.create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        teacher_predictions = []
        student_predictions = []
        teacher_embeddings1 = []
        teacher_embeddings2 = []
        student_embeddings1 = []
        student_embeddings2 = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing distillation metrics", leave=False):
                # Teacher predictions
                teacher_outputs = teacher_model.generate_targets(
                    batch["sentence1"],
                    batch["sentence2"],
                    batch_size=len(batch["sentence1"])
                )
                
                # Student predictions
                student_outputs = student_model.encode_pairs(
                    batch["sentence1"],
                    batch["sentence2"],
                    batch_size=len(batch["sentence1"])
                )
                
                # Collect predictions
                teacher_predictions.extend(teacher_outputs["similarities"].cpu().numpy())
                student_predictions.extend(student_outputs["similarities"].cpu().numpy())
                
                # Collect embeddings
                teacher_embeddings1.extend(teacher_outputs["embeddings1"].cpu().numpy())
                teacher_embeddings2.extend(teacher_outputs["embeddings2"].cpu().numpy())
                student_embeddings1.extend(student_outputs["embeddings1"].cpu().numpy())
                student_embeddings2.extend(student_outputs["embeddings2"].cpu().numpy())
        
        # Convert to arrays
        teacher_predictions = np.array(teacher_predictions)
        student_predictions = np.array(student_predictions)
        teacher_embeddings1 = np.array(teacher_embeddings1)
        teacher_embeddings2 = np.array(teacher_embeddings2)
        student_embeddings1 = np.array(student_embeddings1)
        student_embeddings2 = np.array(student_embeddings2)
        
        # Compute distillation metrics
        metrics = {}
        
        # Similarity transfer quality
        similarity_correlation = self.metrics.compute_similarity_metrics(
            student_predictions, teacher_predictions
        )
        metrics["similarity_transfer"] = similarity_correlation
        
        # Embedding transfer quality
        embedding_correlation1 = np.corrcoef(
            teacher_embeddings1.flatten(),
            student_embeddings1.flatten()
        )[0, 1] if teacher_embeddings1.size > 0 else 0.0
        
        embedding_correlation2 = np.corrcoef(
            teacher_embeddings2.flatten(),
            student_embeddings2.flatten()
        )[0, 1] if teacher_embeddings2.size > 0 else 0.0
        
        metrics["embedding_transfer"] = {
            "sentence1_correlation": embedding_correlation1,
            "sentence2_correlation": embedding_correlation2,
            "average_correlation": (embedding_correlation1 + embedding_correlation2) / 2
        }
        
        # Knowledge retention (how well student retains teacher knowledge)
        retention_score = np.mean(np.abs(teacher_predictions - student_predictions))
        metrics["knowledge_retention"] = 1.0 - retention_score  # Higher is better
        
        return metrics
    
    def _create_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of model comparison"""
        summary = {
            "models": list(comparison_results.keys()),
            "metrics_comparison": {},
            "best_model": {}
        }
        
        # Compare key metrics
        key_metrics = ["spearman", "pearson", "accuracy", "f1"]
        
        for metric in key_metrics:
            metric_values = {}
            
            for model_name, results in comparison_results.items():
                if "similarity_metrics" in results and metric in results["similarity_metrics"]:
                    metric_values[model_name] = results["similarity_metrics"][metric]
            
            if metric_values:
                summary["metrics_comparison"][metric] = metric_values
                
                # Find best model for this metric
                best_model = max(metric_values, key=metric_values.get)
                summary["best_model"][metric] = {
                    "model": best_model,
                    "value": metric_values[best_model]
                }
        
        return summary
    
    def _move_batch_to_device(self, batch: Dict[str, Any], device: str) -> Dict[str, Any]:
        """Move batch to specified device"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value
        
        return device_batch