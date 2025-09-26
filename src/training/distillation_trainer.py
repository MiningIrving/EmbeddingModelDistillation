"""
Main trainer class for embedding model distillation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import json
import numpy as np
from loguru import logger
import wandb

from ..models import TeacherModel, StudentModel
from ..data import CSTSDataset, CSTSDataLoader
from ..evaluation import EvaluationMetrics
from .losses import DistillationLoss
from .utils import save_checkpoint, load_checkpoint


class DistillationTrainer:
    """
    Trainer for embedding model distillation
    """
    
    def __init__(
        self,
        teacher_model: TeacherModel,
        student_model: StudentModel,
        train_dataset: CSTSDataset,
        eval_dataset: Optional[CSTSDataset] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize distillation trainer
        
        Args:
            teacher_model: Teacher model (frozen)
            student_model: Student model (trainable)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config: Training configuration
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}
        
        # Training configuration
        self.output_dir = self.config.get("output_dir", "./output")
        self.num_epochs = self.config.get("num_epochs", 5)
        self.learning_rate = self.config.get("learning_rate", 2e-5)
        self.batch_size = self.config.get("batch_size", 32)
        self.eval_batch_size = self.config.get("eval_batch_size", 64)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.warmup_ratio = self.config.get("warmup_ratio", 0.1)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.save_steps = self.config.get("save_steps", 1000)
        self.eval_steps = self.config.get("eval_steps", 500)
        self.logging_steps = self.config.get("logging_steps", 100)
        self.save_total_limit = self.config.get("save_total_limit", 3)
        
        # Distillation configuration
        distill_config = self.config.get("distillation", {})
        self.temperature = distill_config.get("temperature", 4.0)
        self.alpha = distill_config.get("alpha", 0.7)
        self.beta = distill_config.get("beta", 0.3)
        self.loss_type = distill_config.get("loss_type", "mse")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize training components
        self._setup_training()
        
        # Initialize evaluation metrics
        self.evaluator = EvaluationMetrics()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = -float('inf')
        self.training_history = []
        
        logger.info("Distillation trainer initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Training examples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation examples: {len(self.eval_dataset)}")
    
    def _setup_training(self):
        """Setup training components"""
        # Create data loaders
        self.train_dataloader = CSTSDataLoader.create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            drop_last=self.config.get("dataloader_drop_last", False)
        )
        
        if self.eval_dataset:
            self.eval_dataloader = CSTSDataLoader.create_dataloader(
                self.eval_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=self.config.get("num_workers", 4),
                drop_last=False
            )
        
        # Initialize loss function
        self.loss_fn = DistillationLoss(
            temperature=self.temperature,
            alpha=self.alpha,
            beta=self.beta,
            loss_type=self.loss_type
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Calculate total training steps
        total_steps = len(self.train_dataloader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting training...")
        
        # Initialize wandb if configured
        if self.config.get("logging", {}).get("wandb", {}).get("enabled", False):
            self._init_wandb()
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train one epoch
            train_metrics = self._train_epoch()
            
            # Evaluate if dataset available
            eval_metrics = {}
            if self.eval_dataset:
                eval_metrics = self._evaluate()
            
            # Log epoch results
            epoch_results = {
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics
            }
            self.training_history.append(epoch_results)
            
            # Save checkpoint
            if eval_metrics:
                current_metric = eval_metrics.get("spearman", 0.0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self._save_best_model()
            
            # Log to wandb
            if hasattr(self, 'wandb_run'):
                wandb.log({
                    "epoch": epoch + 1,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"eval_{k}": v for k, v in eval_metrics.items()}
                })
        
        # Save final model
        self._save_final_model()
        
        # Close wandb run
        if hasattr(self, 'wandb_run'):
            wandb.finish()
        
        logger.info("Training completed!")
        
        return {
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "final_model_path": os.path.join(self.output_dir, "final_model")
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass - student
            student_outputs = self.student_model(
                input_ids1=batch["input_ids1"],
                attention_mask1=batch["attention_mask1"],
                input_ids2=batch["input_ids2"],
                attention_mask2=batch["attention_mask2"]
            )
            
            # Forward pass - teacher (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model.generate_targets(
                    batch["sentence1"],
                    batch["sentence2"],
                    batch_size=len(batch["sentence1"])
                )
            
            # Compute loss
            labels = batch.get("labels", None)
            losses = self.loss_fn(student_outputs, teacher_outputs, labels)
            
            loss = losses["total_loss"]
            
            # Backward pass
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss.item()
            num_batches += 1
            
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                self._log_training_step(losses, batch_idx)
            
            # Evaluation
            if self.eval_dataset and self.global_step % self.eval_steps == 0:
                eval_metrics = self._evaluate()
                self.student_model.train()  # Back to training mode
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint()
        
        # Calculate average metrics
        avg_metrics = {
            "loss": total_loss / num_batches
        }
        for key, value in loss_components.items():
            avg_metrics[key] = value / num_batches
        
        return avg_metrics
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        if not self.eval_dataset:
            return {}
        
        logger.info("Running evaluation...")
        
        self.student_model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.student_model(
                    input_ids1=batch["input_ids1"],
                    attention_mask1=batch["attention_mask1"],
                    input_ids2=batch["input_ids2"], 
                    attention_mask2=batch["attention_mask2"]
                )
                
                # Collect predictions and labels
                predictions = outputs["cosine_similarities"].cpu().numpy()
                labels = batch["labels"].cpu().numpy() if "labels" in batch else None
                
                all_predictions.extend(predictions)
                if labels is not None:
                    all_labels.extend(labels)
        
        # Compute metrics
        metrics = {}
        if all_labels:
            metrics = self.evaluator.compute_similarity_metrics(
                all_predictions,
                all_labels
            )
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.student_model.device)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _log_training_step(self, losses: Dict[str, torch.Tensor], batch_idx: int):
        """Log training step metrics"""
        log_msg = f"Step {self.global_step}: "
        for key, value in losses.items():
            log_msg += f"{key}={value.item():.4f} "
        
        logger.debug(log_msg)
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save student model
        self.student_model.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config
        }
        
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _save_best_model(self):
        """Save best model"""
        best_model_dir = os.path.join(self.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        self.student_model.save_pretrained(best_model_dir)
        
        # Save metrics
        with open(os.path.join(best_model_dir, "metrics.json"), "w") as f:
            json.dump({"best_metric": self.best_metric}, f, indent=2)
        
        logger.info(f"Best model saved with metric: {self.best_metric:.4f}")
    
    def _save_final_model(self):
        """Save final model after training"""
        final_model_dir = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        
        self.student_model.save_pretrained(final_model_dir)
        
        # Save training history
        with open(os.path.join(final_model_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Final model saved")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space"""
        checkpoint_dirs = []
        
        for item in os.listdir(self.output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append((
                    int(item.split("-")[1]),
                    os.path.join(self.output_dir, item)
                ))
        
        # Keep only recent checkpoints
        checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
        
        for _, checkpoint_dir in checkpoint_dirs[self.save_total_limit:]:
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.debug(f"Removed old checkpoint: {checkpoint_dir}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_config = self.config.get("logging", {}).get("wandb", {})
        
        self.wandb_run = wandb.init(
            project=wandb_config.get("project", "embedding-distillation"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name"),
            config=self.config,
            reinit=True
        )
        
        logger.info("Weights & Biases logging initialized")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.student_model.device)
            
            self.global_step = training_state["global_step"]
            self.current_epoch = training_state["current_epoch"]
            self.best_metric = training_state["best_metric"]
            
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            self.scheduler.load_state_dict(training_state["scheduler_state"])
            
            logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
        
        # Load student model
        self.student_model.model = self.student_model.model.from_pretrained(checkpoint_path)
        self.student_model.model = self.student_model.model.to(self.student_model.device)