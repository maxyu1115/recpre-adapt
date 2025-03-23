from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from recpre_adapt.data_loaders import AsyncDataLoaderBase, StreamingDataset, collate_fn


def get_streaming_dataloader(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: str, batch_size: int, seq_len: int):

    # Load MathPile dataset
    data_stream = load_dataset("GAIR/MathPile", split=split, streaming=True)

    # Tokenization function (handling missing 'text' fields)
    def tokenize_function(examples):
        if "text" not in examples:
            return {}  # Skip invalid entries
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=seq_len)

    # Tokenize streamed dataset
    tokenized_data = data_stream.map(tokenize_function)

    train_dataset = StreamingDataset(tokenized_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader


class MathPilePMD(AsyncDataLoaderBase):
    def __init__(self, device, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int, seq_len: int):
        super().__init__(
            device,
            {
                "train": get_streaming_dataloader(tokenizer, "train", batch_size, seq_len),
                "validation": get_streaming_dataloader(tokenizer, "validation", batch_size, seq_len)
            }
        )
