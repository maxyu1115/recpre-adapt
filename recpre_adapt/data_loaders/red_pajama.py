import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
import concurrent.futures
import queue


# Custom IterableDataset for PyTorch
class StreamingDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for example in self.dataset:
            if "input_ids" in example:  # Ensure valid tokenized data
                yield example

# Create PyTorch DataLoader
def collate_fn(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}  # Convert list of dicts into batched format

def get_streaming_train_dataloader(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int, seq_len: int):

    # Load RedPajama dataset (default mix)
    redpajama_stream = load_dataset("togethercomputer/RedPajama-Data-1T", "default", split="train", streaming=True)

    # Tokenization function (handling missing 'text' fields)
    def tokenize_function(examples):
        if "text" not in examples:
            return {}  # Skip invalid entries
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=seq_len)

    # Tokenize streamed dataset
    tokenized_pajama = redpajama_stream.map(tokenize_function)

    train_dataset = StreamingDataset(tokenized_pajama)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader


class RedPajamaPMD:
    def __init__(self, device, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_dataloader = get_streaming_train_dataloader(tokenizer, batch_size, seq_len)
        self.device = device

        # Initialize iterator
        self.train_iterator = iter(self.train_dataloader)
        
        # Collect test batches from train_dataloader
        # self.val_dataset = []
        # desired_batches = 10000 // batch_size  # Number of full batches to collect
        # for _ in tqdm(range(desired_batches)):
        #     batch = next(self.train_iterator)
        #     self.val_dataset.append(
        #         torch.tensor(batch["input_ids"], dtype=torch.int64)
        #     )
            
        # Setup for async batch loading
        self.batch_queue = queue.Queue(maxsize=1)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Preload first batch
        self.future = self.executor.submit(self._load_next_batch)

    def _preload_next_batch(self):
        """Submit task to load the next batch asynchronously"""
        self.future = self.executor.submit(self._load_next_batch)
    
    def _load_next_batch(self):
        """Worker function to load the next batch"""
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            batch = next(self.train_iterator)
        
        x = torch.tensor(batch["input_ids"], dtype=torch.int64)
        # Create y by shifting x one position left
        y = x[:, 1:]
        x = x[:, :-1]
        return x, y

    def get_batch(self, split: str):
        if split == "train":
            # Get the preloaded batch
            x, y = self.future.result()

            # Start loading the next batch
            self._preload_next_batch()
            return x.to(self.device), y.to(self.device)
        else:
            raise ValueError(f"Invalid split: {split}")
            
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
