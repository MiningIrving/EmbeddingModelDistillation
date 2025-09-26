"""
Data utilities for downloading and preprocessing CSTS dataset
"""

import os
import json
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from loguru import logger
import zipfile
from pathlib import Path


def download_csts_data(data_dir: str = "./data/csts") -> bool:
    """
    Download CSTS (Chinese STS-B) dataset from GitHub
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        bool: Success status
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # CSTS dataset URLs (using publicly available Chinese STS data)
    urls = {
        "train": "https://raw.githubusercontent.com/zejunwang1/CSTS/main/data/train.txt",
        "dev": "https://raw.githubusercontent.com/zejunwang1/CSTS/main/data/dev.txt", 
        "test": "https://raw.githubusercontent.com/zejunwang1/CSTS/main/data/test.txt"
    }
    
    try:
        for split, url in urls.items():
            file_path = os.path.join(data_dir, f"{split}.txt")
            if os.path.exists(file_path):
                logger.info(f"{split} file already exists, skipping download")
                continue
                
            logger.info(f"Downloading {split} data from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.info(f"Successfully downloaded {split} data")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading CSTS data: {e}")
        return False


def preprocess_csts_data(data_dir: str = "./data/csts") -> bool:
    """
    Preprocess CSTS data files into JSONL format
    
    Args:
        data_dir: Directory containing the raw data files
        
    Returns:
        bool: Success status
    """
    try:
        for split in ["train", "dev", "test"]:
            input_file = os.path.join(data_dir, f"{split}.txt")
            output_file = os.path.join(data_dir, f"{split}.jsonl")
            
            if not os.path.exists(input_file):
                logger.warning(f"Input file {input_file} does not exist, skipping")
                continue
                
            if os.path.exists(output_file):
                logger.info(f"Output file {output_file} already exists, skipping")
                continue
            
            logger.info(f"Preprocessing {split} data")
            
            with open(input_file, 'r', encoding='utf-8') as f_in, \
                 open(output_file, 'w', encoding='utf-8') as f_out:
                
                for line_num, line in enumerate(f_in):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Expected format: sentence1 \t sentence2 \t score
                    parts = line.split('\t')
                    if len(parts) < 3:
                        logger.warning(f"Skipping malformed line {line_num + 1} in {split}")
                        continue
                    
                    sentence1 = parts[0].strip()
                    sentence2 = parts[1].strip()
                    try:
                        score = float(parts[2].strip())
                    except ValueError:
                        logger.warning(f"Invalid score in line {line_num + 1} in {split}")
                        continue
                    
                    # Create JSON record
                    record = {
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                        "score": score,
                        "split": split
                    }
                    
                    f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Successfully preprocessed {split} data")
        
        return True
        
    except Exception as e:
        logger.error(f"Error preprocessing CSTS data: {e}")
        return False


def create_sample_data(data_dir: str = "./data/csts") -> bool:
    """
    Create sample CSTS data for testing when real data is not available
    
    Args:
        data_dir: Directory to save sample data
        
    Returns:
        bool: Success status
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample Chinese sentence pairs with similarity scores
    sample_data = {
        "train": [
            {"sentence1": "今天天气很好", "sentence2": "今日天气不错", "score": 4.5},
            {"sentence1": "我喜欢吃苹果", "sentence2": "我爱吃水果", "score": 3.2},
            {"sentence1": "北京是中国的首都", "sentence2": "中国的首都是北京", "score": 5.0},
            {"sentence1": "这本书很有趣", "sentence2": "这部电影很精彩", "score": 1.8},
            {"sentence1": "学习中文很重要", "sentence2": "掌握汉语非常关键", "score": 4.2},
        ] * 100,  # Repeat to create more samples
        
        "dev": [
            {"sentence1": "春天来了", "sentence2": "春季到了", "score": 4.8},
            {"sentence1": "我在学校", "sentence2": "我在公司", "score": 2.1},
            {"sentence1": "明天下雨", "sentence2": "明日会下雨", "score": 4.9},
        ] * 20,
        
        "test": [
            {"sentence1": "猫在睡觉", "sentence2": "小猫在休息", "score": 4.0},
            {"sentence1": "开车去上班", "sentence2": "骑车去工作", "score": 3.5},
            {"sentence1": "读书是好习惯", "sentence2": "阅读是个好习惯", "score": 4.7},
        ] * 20
    }
    
    try:
        for split, data in sample_data.items():
            output_file = os.path.join(data_dir, f"{split}.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, record in enumerate(data):
                    record["split"] = split
                    record["id"] = f"{split}_{i}"
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Created sample {split} data with {len(data)} examples")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False


def load_dataset_stats(data_dir: str = "./data/csts") -> Dict[str, Any]:
    """
    Load and return dataset statistics
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Dict containing dataset statistics
    """
    stats = {}
    
    for split in ["train", "dev", "test"]:
        file_path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        stats[split] = {
            "num_examples": len(lines),
            "file_size": os.path.getsize(file_path)
        }
        
        # Calculate score statistics
        scores = []
        for line in lines:
            try:
                data = json.loads(line)
                scores.append(data["score"])
            except:
                continue
                
        if scores:
            stats[split].update({
                "score_mean": sum(scores) / len(scores),
                "score_min": min(scores),
                "score_max": max(scores)
            })
    
    return stats