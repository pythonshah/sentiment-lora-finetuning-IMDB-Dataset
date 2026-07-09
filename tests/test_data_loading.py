"""
tests/test_data_loading.py
---------------------------
Unit tests for the CSV loading, label mapping, and train/eval split logic in train.py.
Uses a small synthetic CSV so tests run instantly without needing the real IMDB dataset.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import load_data


@pytest.fixture
def sample_csv(tmp_path):
    data = {
        "review": [f"review number {i}" for i in range(20)],
        "sentiment": (["positive", "negative"] * 10),
    }
    df = pd.DataFrame(data)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_data_basic_split(sample_csv):
    train_df, eval_df = load_data(sample_csv, sample_size=-1, test_size=0.2, seed=42)
    assert len(train_df) + len(eval_df) == 20
    assert "label" in train_df.columns
    assert set(train_df["label"].unique()).issubset({0, 1})


def test_load_data_sampling(sample_csv):
    train_df, eval_df = load_data(sample_csv, sample_size=10, test_size=0.2, seed=42)
    assert len(train_df) + len(eval_df) == 10


def test_load_data_missing_columns_raises(tmp_path):
    bad_df = pd.DataFrame({"text": ["a"], "label_name": ["positive"]})
    path = tmp_path / "bad.csv"
    bad_df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(str(path), sample_size=-1, test_size=0.2, seed=42)


def test_load_data_reset_index(sample_csv):
    train_df, eval_df = load_data(sample_csv, sample_size=-1, test_size=0.2, seed=42)
    assert list(train_df.index) == list(range(len(train_df)))
    assert list(eval_df.index) == list(range(len(eval_df)))
