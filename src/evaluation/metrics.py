"""
Evaluation metrics for embedding models
"""

import numpy as np
import torch
from typing import List, Dict, Any, Union, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from loguru import logger


class EvaluationMetrics:
    """
    Evaluation metrics for embedding model similarity tasks
    """
    
    def __init__(self):
        """Initialize evaluation metrics"""
        pass
    
    def compute_similarity_metrics(
        self,
        predictions: Union[List[float], np.ndarray, torch.Tensor],
        labels: Union[List[float], np.ndarray, torch.Tensor],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute similarity evaluation metrics
        
        Args:
            predictions: Predicted similarity scores
            labels: Ground truth similarity scores
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(labels))
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        
        if len(predictions) == 0:
            logger.warning("No valid predictions for evaluation")
            return {}
        
        metrics = {}
        
        # Correlation metrics
        try:
            pearson_corr, pearson_p = pearsonr(predictions, labels)
            metrics["pearson"] = pearson_corr
            metrics["pearson_pvalue"] = pearson_p
        except Exception as e:
            logger.warning(f"Error computing Pearson correlation: {e}")
            metrics["pearson"] = 0.0
        
        try:
            spearman_corr, spearman_p = spearmanr(predictions, labels)
            metrics["spearman"] = spearman_corr
            metrics["spearman_pvalue"] = spearman_p
        except Exception as e:
            logger.warning(f"Error computing Spearman correlation: {e}")
            metrics["spearman"] = 0.0
        
        # Regression metrics
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        rmse = np.sqrt(mse)
        
        metrics.update({
            "mse": mse,
            "mae": mae,
            "rmse": rmse
        })
        
        # Binary classification metrics (using threshold)
        binary_predictions = (predictions > threshold).astype(int)
        binary_labels = (labels > threshold).astype(int)
        
        try:
            accuracy = accuracy_score(binary_labels, binary_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                binary_labels, binary_predictions, average='binary', zero_division=0
            )
            
            metrics.update({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
        except Exception as e:
            logger.warning(f"Error computing classification metrics: {e}")
        
        return metrics
    
    def compute_ranking_metrics(
        self,
        similarities: Union[List[float], np.ndarray],
        labels: Union[List[float], np.ndarray],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute ranking metrics
        
        Args:
            similarities: Predicted similarity scores
            labels: Ground truth similarity scores
            k_values: K values for top-k metrics
            
        Returns:
            Dictionary of ranking metrics
        """
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Get ranking indices (descending order)
        pred_indices = np.argsort(similarities)[::-1]
        true_indices = np.argsort(labels)[::-1]
        
        metrics = {}
        
        # Compute top-k accuracy
        for k in k_values:
            if k <= len(similarities):
                top_k_pred = set(pred_indices[:k])
                top_k_true = set(true_indices[:k])
                
                top_k_accuracy = len(top_k_pred.intersection(top_k_true)) / k
                metrics[f"top_{k}_accuracy"] = top_k_accuracy
        
        # Compute normalized discounted cumulative gain (NDCG)
        try:
            ndcg = self._compute_ndcg(similarities, labels)
            metrics["ndcg"] = ndcg
        except Exception as e:
            logger.warning(f"Error computing NDCG: {e}")
            metrics["ndcg"] = 0.0
        
        return metrics
    
    def _compute_ndcg(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        k: Optional[int] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG)
        
        Args:
            predictions: Predicted scores
            labels: Ground truth relevance scores
            k: Number of top results to consider
            
        Returns:
            NDCG score
        """
        if k is None:
            k = len(predictions)
        
        # Sort by predictions (descending)
        sorted_indices = np.argsort(predictions)[::-1][:k]
        sorted_labels = labels[sorted_indices]
        
        # Compute DCG
        dcg = 0.0
        for i, relevance in enumerate(sorted_labels):
            dcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        # Compute IDCG (perfect ranking)
        ideal_sorted_labels = np.sort(labels)[::-1][:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_sorted_labels):
            idcg += (2 ** relevance - 1) / np.log2(i + 2)
        
        # Return NDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def compute_embedding_quality_metrics(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: Union[List[int], np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute embedding quality metrics
        
        Args:
            embeddings: Embedding vectors
            labels: Optional labels for supervised metrics
            
        Returns:
            Dictionary of embedding quality metrics
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        embeddings = np.array(embeddings)
        metrics = {}
        
        # Basic statistics
        metrics["embedding_dim"] = embeddings.shape[1]
        metrics["num_samples"] = embeddings.shape[0]
        metrics["mean_norm"] = np.mean(np.linalg.norm(embeddings, axis=1))
        metrics["std_norm"] = np.std(np.linalg.norm(embeddings, axis=1))
        
        # Cosine similarity statistics
        similarity_matrix = np.dot(embeddings, embeddings.T)
        norm_matrix = np.outer(
            np.linalg.norm(embeddings, axis=1),
            np.linalg.norm(embeddings, axis=1)
        )
        cosine_similarities = similarity_matrix / (norm_matrix + 1e-8)
        
        # Remove diagonal (self-similarities)
        mask = ~np.eye(cosine_similarities.shape[0], dtype=bool)
        off_diagonal_similarities = cosine_similarities[mask]
        
        metrics.update({
            "mean_cosine_similarity": np.mean(off_diagonal_similarities),
            "std_cosine_similarity": np.std(off_diagonal_similarities),
            "min_cosine_similarity": np.min(off_diagonal_similarities),
            "max_cosine_similarity": np.max(off_diagonal_similarities)
        })
        
        # If labels are provided, compute supervised metrics
        if labels is not None:
            labels = np.array(labels)
            
            # Intra-class vs inter-class similarities
            intra_similarities = []
            inter_similarities = []
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = cosine_similarities[i, j]
                    
                    if labels[i] == labels[j]:
                        intra_similarities.append(similarity)
                    else:
                        inter_similarities.append(similarity)
            
            if intra_similarities and inter_similarities:
                metrics.update({
                    "mean_intra_class_similarity": np.mean(intra_similarities),
                    "mean_inter_class_similarity": np.mean(inter_similarities),
                    "similarity_gap": np.mean(intra_similarities) - np.mean(inter_similarities)
                })
        
        return metrics


def compute_similarity_metrics(
    predictions: Union[List[float], np.ndarray, torch.Tensor],
    labels: Union[List[float], np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Convenience function to compute similarity metrics
    
    Args:
        predictions: Predicted similarity scores
        labels: Ground truth similarity scores
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    evaluator = EvaluationMetrics()
    return evaluator.compute_similarity_metrics(predictions, labels, threshold)


def compute_correlation_metrics(
    predictions: Union[List[float], np.ndarray],
    labels: Union[List[float], np.ndarray]
) -> Tuple[float, float]:
    """
    Compute Pearson and Spearman correlations
    
    Args:
        predictions: Predicted values
        labels: Ground truth values
        
    Returns:
        Tuple of (pearson_correlation, spearman_correlation)
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(labels))
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    if len(predictions) < 2:
        return 0.0, 0.0
    
    try:
        pearson_corr, _ = pearsonr(predictions, labels)
    except:
        pearson_corr = 0.0
    
    try:
        spearman_corr, _ = spearmanr(predictions, labels)
    except:
        spearman_corr = 0.0
    
    return pearson_corr, spearman_corr