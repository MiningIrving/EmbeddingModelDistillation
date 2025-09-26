"""
Base embedding model class
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig
from loguru import logger
import numpy as np


class EmbeddingModel(nn.Module):
    """
    Base class for embedding models used in distillation
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
        device: str = "auto",
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize embedding model
        
        Args:
            model_name_or_path: Path or name of the pretrained model
            max_length: Maximum sequence length
            device: Device to load model on
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading
        """
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model components
        self._load_model(**kwargs)
        
        logger.info(f"Loaded embedding model: {model_name_or_path}")
        logger.info(f"Model device: {self.device}")
        logger.info(f"Model parameters: {self.get_parameter_count():,}")
    
    def _load_model(self, **kwargs):
        """Load model, tokenizer, and config"""
        try:
            # Load config
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float32,
                **kwargs
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name_or_path}: {e}")
            raise
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        return_tensors: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode sentences into embeddings
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings
            return_tensors: Whether to return torch tensors
            **kwargs: Additional arguments
            
        Returns:
            Embeddings as tensor or numpy array
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_embeddings = self._encode_batch(
                batch_sentences,
                normalize_embeddings=normalize_embeddings,
                **kwargs
            )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if not return_tensors:
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def _encode_batch(
        self,
        sentences: List[str],
        normalize_embeddings: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode a batch of sentences
        
        Args:
            sentences: List of sentences to encode
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional arguments
            
        Returns:
            Batch embeddings tensor
        """
        # Tokenize
        inputs = self.tokenizer(
            sentences,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get embeddings (using mean pooling of last hidden states)
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )
        
        # Normalize if requested
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_method: str = "mean"
    ) -> torch.Tensor:
        """
        Pool token embeddings to sentence embeddings
        
        Args:
            hidden_states: Token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            pooling_method: Pooling method ('mean', 'cls', 'max')
            
        Returns:
            Sentence embeddings [batch_size, hidden_size]
        """
        if pooling_method == "mean":
            # Mean pooling with attention mask
            masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
            embeddings = sum_embeddings / sum_mask
            
        elif pooling_method == "cls":
            # Use [CLS] token embedding (first token)
            embeddings = hidden_states[:, 0, :]
            
        elif pooling_method == "max":
            # Max pooling
            masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
            masked_embeddings[attention_mask == 0] = -torch.inf
            embeddings = torch.max(masked_embeddings, dim=1)[0]
            
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing embeddings and other outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Pool embeddings
        embeddings = self._pool_embeddings(
            outputs.last_hidden_state,
            attention_mask
        )
        
        return {
            "embeddings": embeddings,
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": getattr(outputs, "pooler_output", None)
        }
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_parameters(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self):
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")
    
    def similarity(
        self,
        sentences1: Union[str, List[str]],
        sentences2: Union[str, List[str]],
        **kwargs
    ) -> Union[float, List[float]]:
        """
        Compute cosine similarity between sentence pairs
        
        Args:
            sentences1: First sentences
            sentences2: Second sentences
            **kwargs: Additional arguments
            
        Returns:
            Similarity scores
        """
        # Ensure both inputs are lists
        if isinstance(sentences1, str):
            sentences1 = [sentences1]
        if isinstance(sentences2, str):
            sentences2 = [sentences2]
        
        # Encode sentences
        embeddings1 = self.encode(sentences1, **kwargs)
        embeddings2 = self.encode(sentences2, **kwargs)
        
        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            embeddings1, embeddings2, dim=1
        )
        
        # Convert to list if multiple similarities
        if len(similarities) == 1:
            return similarities.item()
        else:
            return similarities.cpu().tolist()