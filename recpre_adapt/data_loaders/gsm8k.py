import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import random


class GSM8KDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Format as "Question: {question}\nAnswer: {answer}"
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        
        # Tokenize with padding and truncation
        encodings = self.tokenizer(text, truncation=True, 
                                  padding="max_length", 
                                  max_length=self.seq_len,
                                  return_tensors="pt")
        
        # Remove batch dimension added by return_tensors="pt"
        input_ids = encodings["input_ids"].squeeze(0)
        
        # Create target by shifting input_ids
        target_ids = input_ids[1:].clone()
        input_ids = input_ids[:-1]
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    return input_ids, target_ids


class GSM8K:
    def __init__(self, device, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
                 batch_size: int, seq_len: int, val_split: float = 0.1):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        # Split train into train and validation
        train_data = load_dataset("openai/gsm8k", "main", split="train")
        
        # Determine split indices
        val_size = int(len(train_data) * val_split) # type: ignore
        train_size = len(train_data) - val_size # type: ignore
        
        # Create random indices for splitting
        indices = list(range(len(train_data))) # type: ignore
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create train and validation datasets
        train_dataset = GSM8KDataset(
            [train_data[i] for i in train_indices],  # type: ignore
            tokenizer, 
            seq_len
        )
        val_dataset = GSM8KDataset(
            [train_data[i] for i in val_indices],  # type: ignore
            tokenizer, 
            seq_len
        )
        
        # Create test dataset
        test_dataset = GSM8KDataset(load_dataset("openai/gsm8k", "main", split="test"), tokenizer, seq_len)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Initialize iterators
        self.train_iterator = iter(self.train_dataloader)
        self.val_iterator = iter(self.val_dataloader)
        self.test_iterator = iter(self.test_dataloader)

    def get_batch(self, split: str):
        if split == "train":
            try:
                x, y = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_dataloader)
                x, y = next(self.train_iterator)
        elif split == "val":
            try:
                x, y = next(self.val_iterator)
            except StopIteration:
                self.val_iterator = iter(self.val_dataloader)
                x, y = next(self.val_iterator)
        elif split == "test":
            try:
                x, y = next(self.test_iterator)
            except StopIteration:
                self.test_iterator = iter(self.test_dataloader)
                x, y = next(self.test_iterator)
        else:
            raise ValueError(f"Invalid split: {split}")
            
        return x.to(self.device), y.to(self.device)
