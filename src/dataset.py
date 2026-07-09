"""
dataset.py
----------
Custom PyTorch Dataset for IMDB-style sentiment classification data.
"""

import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    """
    Wraps tokenized text + labels into a PyTorch Dataset compatible
    with HuggingFace's Trainer API.

    Args:
        texts (list[str]): List of raw review texts.
        labels (list[int]): List of integer labels (0 = negative, 1 = positive).
        tokenizer: A HuggingFace tokenizer instance.
        max_length (int): Max token sequence length (default: 256).
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
