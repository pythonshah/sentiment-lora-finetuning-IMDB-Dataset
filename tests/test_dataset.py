"""
tests/test_dataset.py
----------------------
Unit tests for the SentimentDataset class. Uses a lightweight fake tokenizer
so tests run instantly with no model downloads.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import SentimentDataset


class FakeTokenizer:
    """Minimal tokenizer stand-in that mimics the HF tokenizer call interface."""

    def __call__(self, text, padding=None, truncation=None, max_length=256, return_tensors=None):
        ids = [ord(c) % 100 for c in text[:max_length]]
        ids += [0] * (max_length - len(ids))
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([[1] * len(ids)]),
        }


def test_dataset_length():
    texts = ["good movie", "bad movie", "okay movie"]
    labels = [1, 0, 1]
    ds = SentimentDataset(texts, labels, FakeTokenizer(), max_length=16)
    assert len(ds) == 3


def test_dataset_item_shape():
    texts = ["great acting and story"]
    labels = [1]
    ds = SentimentDataset(texts, labels, FakeTokenizer(), max_length=16)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 16
    assert item["labels"].item() == 1


def test_dataset_label_dtype():
    ds = SentimentDataset(["a review"], [0], FakeTokenizer(), max_length=8)
    item = ds[0]
    assert item["labels"].dtype == torch.long


def test_empty_dataset():
    ds = SentimentDataset([], [], FakeTokenizer(), max_length=8)
    assert len(ds) == 0
