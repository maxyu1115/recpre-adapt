from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm

from recpre_adapt.data_loaders import AsyncDataLoaderBase, StreamingDataset, collate_fn

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


class RedPajamaPMD(AsyncDataLoaderBase):
    def __init__(self, device, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int, seq_len: int):
        super().__init__(
            device,
            {
                "train": get_streaming_train_dataloader(tokenizer, batch_size, seq_len)
            }
        )

        # Collect test batches from train_dataloader
        # self.val_dataset = []
        # desired_batches = 10000 // batch_size  # Number of full batches to collect
        # for _ in tqdm(range(desired_batches)):
        #     batch = next(self.train_iterator)
        #     self.val_dataset.append(
        #         torch.tensor(batch["input_ids"], dtype=torch.int64)
        #     )
