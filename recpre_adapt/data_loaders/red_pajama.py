import random
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
import torch

from recpre_adapt.data_loaders import PoorMansDataLoaderBase, StreamingDataset, collate_fn

def get_streaming_train_dataloader(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int, seq_len: int):

    # Load RedPajama dataset (default mix)
    redpajama_stream = load_dataset("togethercomputer/RedPajama-Data-1T", "default", split="train", streaming=True)

    # Tokenization function (handling missing 'text' fields)
    def tokenize_function(examples):
        if "text" not in examples:
            return {}  # Skip invalid entries
        result = tokenizer(examples["text"])
        tokens: list[int] = result["input_ids"] # type: ignore
        tokens_length = len(tokens)
        # Skip if text is too short. TODO: This is a temporary hack, should actually pad
        if tokens_length < seq_len:
            result["input_ids"] = tokens + [tokenizer.pad_token_id] * (seq_len - tokens_length)
            return result
        else:
            random_indice = random.randint(0, tokens_length - seq_len)
            result["input_ids"] = tokens[random_indice:random_indice+seq_len]
            return result

    # Tokenize streamed dataset
    tokenized_pajama = redpajama_stream.map(tokenize_function)

    train_dataset = StreamingDataset(tokenized_pajama)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1, persistent_workers=True, prefetch_factor=2)

    return train_dataloader


class RedPajamaPMD(PoorMansDataLoaderBase):
    def __init__(self, device, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int, seq_len: int):
        super().__init__()
        self.device = device

        self.train_dataloader = get_streaming_train_dataloader(tokenizer, batch_size, seq_len)
        self.train_iterator = iter(self.train_dataloader)

        # Collect test batches from train_dataloader
        # self.val_dataset = []
        # desired_batches = 10000 // batch_size  # Number of full batches to collect
        # for _ in tqdm(range(desired_batches)):
        #     batch = next(self.train_iterator)
        #     self.val_dataset.append(
        #         torch.tensor(batch["input_ids"], dtype=torch.int64)
        #     )

    def _get_batch(self):
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            return next(self.train_iterator)

    def get_batch(self, split: str):
        if split != "train":
            raise ValueError(f"Invalid split: {split}")
        batch = self._get_batch()
        while "input_ids" not in batch:
            batch = self._get_batch()
        tokens = torch.tensor(batch["input_ids"], dtype=torch.int64)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        return x.to(self.device), y.to(self.device)
