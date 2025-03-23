import abc
import concurrent.futures
from typing import Iterator
import torch
from torch.utils.data import IterableDataset, DataLoader

class PoorMansDataLoaderBase(abc.ABC):
    @abc.abstractmethod
    def get_batch(self, split: str):
        raise NotImplementedError("get_batch must be implemented by subclass")


class AsyncDataLoaderBase(PoorMansDataLoaderBase):
    def __init__(self, device: torch.device, dataloaders: dict[str, DataLoader]):
        self.device = device
        self.dataloaders = dataloaders
        self.iterators = {split: iter(dataloader) for split, dataloader in self.dataloaders.items()}
        self.futures: dict[str, concurrent.futures.Future] = dict()

        # Setup for async batch loading
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        for split in self.dataloaders:
            self._preload_next_batch(split)

    def _preload_next_batch(self, split: str):
        """Submit task to load the next batch asynchronously"""
        self.futures[split] = self.executor.submit(self._load_next_batch, split)
    
    def _load_next_batch(self, split: str):
        """Worker function to load the next batch"""
        x = torch.tensor(self.load_batch(split), dtype=torch.int64)
        # Create y by shifting x one position left
        y = x[:, 1:]
        x = x[:, :-1]
        return x, y

    def load_batch(self, split: str):
        if split in self.dataloaders:
            try:
                batch = next(self.iterators[split])
            except StopIteration:
                self.iterators[split] = iter(self.dataloaders[split])
                batch = next(self.iterators[split])
            return batch["input_ids"]
        else:
            raise ValueError(f"Invalid split: {split}")

    def get_batch(self, split: str):
        if split in self.dataloaders:
            # Get the preloaded batch
            x, y = self.futures[split].result()

            # Start loading the next batch
            self._preload_next_batch(split)
            return x.to(self.device), y.to(self.device)
        else:
            raise ValueError(f"Invalid split: {split}")

    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Custom IterableDataset for PyTorch
class StreamingDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for example in self.dataset:
            assert "input_ids" in example, "input_ids not found in example"
            yield example

# Create PyTorch DataLoader
def collate_fn(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}  # Convert list of dicts into batched format

