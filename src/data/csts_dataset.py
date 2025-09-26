"""
CSTS Dataset implementation for embedding model distillation
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from loguru import logger


class CSTSDataset(Dataset):
    """
    Chinese STS-B Dataset for embedding model distillation
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        return_labels: bool = True
    ):
        """
        Initialize CSTS Dataset
        
        Args:
            data_path: Path to the JSONL data file
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            return_labels: Whether to return similarity labels
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_labels = return_labels
        
        # Load data
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        
        if not os.path.exists(self.data_path):
            logger.error(f"Data file {self.data_path} does not exist")
            return data
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    
                    # Validate required fields
                    if 'sentence1' not in record or 'sentence2' not in record:
                        logger.warning(f"Missing sentences in line {line_num + 1}")
                        continue
                    
                    if self.return_labels and 'score' not in record:
                        logger.warning(f"Missing score in line {line_num + 1}")
                        continue
                    
                    data.append(record)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in line {line_num + 1}: {e}")
                    continue
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example"""
        record = self.data[idx]
        
        sentence1 = record['sentence1']
        sentence2 = record['sentence2']
        
        # Prepare the output
        item = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'idx': idx
        }
        
        # Add tokenized inputs if tokenizer is provided
        if self.tokenizer is not None:
            # Tokenize sentences
            inputs1 = self.tokenizer(
                sentence1,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs2 = self.tokenizer(
                sentence2,
                max_length=self.max_length, 
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Remove batch dimension and add to item
            item.update({
                'input_ids1': inputs1['input_ids'].squeeze(0),
                'attention_mask1': inputs1['attention_mask'].squeeze(0),
                'input_ids2': inputs2['input_ids'].squeeze(0),
                'attention_mask2': inputs2['attention_mask'].squeeze(0)
            })
        
        # Add label if available and requested
        if self.return_labels and 'score' in record:
            # Normalize score to [0, 1] range (assuming original range is [0, 5])
            normalized_score = record['score'] / 5.0
            item['label'] = torch.tensor(normalized_score, dtype=torch.float32)
            item['raw_score'] = record['score']
        
        return item
    
    def get_text_pairs(self) -> List[Tuple[str, str]]:
        """Get all text pairs without tokenization"""
        return [(record['sentence1'], record['sentence2']) for record in self.data]
    
    def get_labels(self) -> List[float]:
        """Get all similarity scores"""
        if not self.return_labels:
            return []
        return [record.get('score', 0.0) for record in self.data]


class CSTSDataLoader:
    """
    Data loader factory for CSTS dataset
    """
    
    @staticmethod
    def create_dataloader(
        dataset: CSTSDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader for the CSTS dataset
        
        Args:
            dataset: CSTS dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=CSTSDataLoader._collate_fn
        )
    
    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for batching CSTS data
        
        Args:
            batch: List of dataset items
            
        Returns:
            Batched data
        """
        # Initialize batch dictionary
        batched = {
            'sentence1': [],
            'sentence2': [],
            'idx': []
        }
        
        # Check if tokenized inputs are available
        has_tokens = 'input_ids1' in batch[0]
        has_labels = 'label' in batch[0]
        
        if has_tokens:
            batched.update({
                'input_ids1': [],
                'attention_mask1': [],
                'input_ids2': [],
                'attention_mask2': []
            })
        
        if has_labels:
            batched.update({
                'labels': [],
                'raw_scores': []
            })
        
        # Collect data from batch
        for item in batch:
            batched['sentence1'].append(item['sentence1'])
            batched['sentence2'].append(item['sentence2'])
            batched['idx'].append(item['idx'])
            
            if has_tokens:
                batched['input_ids1'].append(item['input_ids1'])
                batched['attention_mask1'].append(item['attention_mask1'])
                batched['input_ids2'].append(item['input_ids2'])
                batched['attention_mask2'].append(item['attention_mask2'])
            
            if has_labels:
                batched['labels'].append(item['label'])
                batched['raw_scores'].append(item['raw_score'])
        
        # Stack tensors
        if has_tokens:
            batched['input_ids1'] = torch.stack(batched['input_ids1'])
            batched['attention_mask1'] = torch.stack(batched['attention_mask1'])
            batched['input_ids2'] = torch.stack(batched['input_ids2'])
            batched['attention_mask2'] = torch.stack(batched['attention_mask2'])
        
        if has_labels:
            batched['labels'] = torch.stack(batched['labels'])
        
        return batched


def create_csts_datasets(
    data_dir: str,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 512,
    splits: List[str] = None
) -> Dict[str, CSTSDataset]:
    """
    Create CSTS datasets for multiple splits
    
    Args:
        data_dir: Directory containing the dataset files
        tokenizer: Tokenizer for text encoding
        max_length: Maximum sequence length
        splits: List of splits to load (default: ['train', 'dev', 'test'])
        
    Returns:
        Dictionary mapping split names to datasets
    """
    if splits is None:
        splits = ['train', 'dev', 'test']
    
    datasets = {}
    
    for split in splits:
        data_path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(data_path):
            datasets[split] = CSTSDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_length=max_length,
                return_labels=True
            )
        else:
            logger.warning(f"Dataset file for {split} split not found: {data_path}")
    
    return datasets