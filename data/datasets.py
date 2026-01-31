"""
Data loading for EDP experiments.

Supports:
- TinyStories: Method validation, routing behavior analysis, ablations
- WikiText-103: Final results, scaling curves
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np


class TextDataset(Dataset):
    """
    Generic text dataset for language modeling.
    
    Handles tokenization and chunking into fixed-length sequences.
    """
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        seq_len: int = 256,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len
        
        # Tokenize all texts
        self.all_tokens = []
        for text in texts:
            if text.strip():  # Skip empty texts
                tokens = tokenizer.encode(text, add_special_tokens=False)
                self.all_tokens.extend(tokens)
        
        # Convert to tensor
        self.all_tokens = torch.tensor(self.all_tokens, dtype=torch.long)
        
        # Calculate number of samples
        total_len = len(self.all_tokens)
        if total_len < seq_len + 1:
            self.n_samples = 0
        else:
            self.n_samples = (total_len - seq_len - 1) // self.stride + 1
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len + 1
        
        chunk = self.all_tokens[start:end]
        
        input_ids = chunk[:-1]
        labels = chunk[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids),
        }


class TinyStoriesDataset(Dataset):
    """
    TinyStories dataset wrapper.
    
    Properties:
    - Short sequences
    - Clear "easy vs hard" tokens
    - Strong signal for adaptive depth
    """
    
    def __init__(
        self,
        split: str = 'train',
        seq_len: int = 256,
        tokenizer_name: str = 'gpt2',
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load dataset
        print(f"Loading TinyStories {split} split...")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir,
        )
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        # Extract texts
        texts = [item['text'] for item in dataset]
        
        # Create base dataset
        self.base_dataset = TextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            seq_len=seq_len,
        )
        
        print(f"Created TinyStories dataset with {len(self.base_dataset)} samples")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]
    
    def get_tokenizer(self):
        return self.tokenizer


class WikiText103Dataset(Dataset):
    """
    WikiText-103 dataset wrapper.
    
    Properties:
    - Natural language entropy
    - Long-range dependencies
    - Standard pretraining benchmark
    """
    
    def __init__(
        self,
        split: str = 'train',
        seq_len: int = 256,
        tokenizer_name: str = 'gpt2',
        cache_dir: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load dataset
        print(f"Loading WikiText-103 {split} split...")
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split=split,
            cache_dir=cache_dir,
        )
        
        # Extract and filter texts
        texts = [item['text'] for item in dataset if item['text'].strip()]
        
        # Create base dataset
        self.base_dataset = TextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            seq_len=seq_len,
        )
        
        print(f"Created WikiText-103 dataset with {len(self.base_dataset)} samples")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]
    
    def get_tokenizer(self):
        return self.tokenizer


def create_dataloaders(
    dataset_name: str,
    batch_size: int,
    seq_len: int = 256,
    tokenizer_name: str = 'gpt2',
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_name: 'tinystories' or 'wikitext103'
        batch_size: Batch size
        seq_len: Sequence length
        tokenizer_name: Tokenizer to use
        num_workers: Number of dataloader workers
        max_train_samples: Limit training samples (for debugging)
        max_val_samples: Limit validation samples
        cache_dir: Cache directory for datasets
        
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'tinystories':
        DatasetClass = TinyStoriesDataset
        # TinyStories only has train/validation splits
        splits = {'train': 'train', 'val': 'validation', 'test': 'validation'}
    elif dataset_name in ['wikitext103', 'wikitext-103']:
        DatasetClass = WikiText103Dataset
        splits = {'train': 'train', 'val': 'validation', 'test': 'test'}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create datasets
    train_dataset = DatasetClass(
        split=splits['train'],
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        max_samples=max_train_samples,
        cache_dir=cache_dir,
    )
    
    val_dataset = DatasetClass(
        split=splits['val'],
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        max_samples=max_val_samples,
        cache_dir=cache_dir,
    )
    
    test_dataset = DatasetClass(
        split=splits['test'],
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        max_samples=max_val_samples,
        cache_dir=cache_dir,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    tokenizer = train_dataset.get_tokenizer()
    
    return train_loader, val_loader, test_loader, tokenizer


def collate_fn(batch):
    """Custom collate function for padding."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }
