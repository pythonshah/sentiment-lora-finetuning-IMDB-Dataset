"""
train.py
--------
Fine-tunes DistilBERT for binary sentiment classification (IMDB reviews)
using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Usage:
    python src/train.py --data_path data/IMDB_Dataset.csv --sample_size 1200 --epochs 3

Author: Adnan
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from dataset import SentimentDataset

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for sentiment classification")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CSV file with 'review' and 'sentiment' columns")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base HuggingFace model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./my_lora_model",
                        help="Directory for training checkpoints/logs")
    parser.add_argument("--save_dir", type=str, default="./my_first_finetuned_model",
                        help="Directory to save the final fine-tuned model")
    parser.add_argument("--sample_size", type=int, default=1200,
                        help="Number of rows to sample from the dataset (use -1 for full dataset)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data reserved for evaluation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Computes accuracy, F1, precision, and recall for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def load_data(data_path: str, sample_size: int, test_size: float, seed: int):
    df = pd.read_csv(data_path)

    if "sentiment" not in df.columns or "review" not in df.columns:
        raise ValueError(
            f"Expected columns 'review' and 'sentiment' in {data_path}, "
            f"found: {list(df.columns)}"
        )

    df = df.dropna(subset=["review", "sentiment"])
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    if df["label"].isna().any():
        raise ValueError("Found sentiment values other than 'positive'/'negative'.")

    if sample_size > 0 and sample_size < len(df):
        df = df.sample(sample_size, random_state=seed)

    train_df, eval_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def build_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.data_path} ...")
    train_df, eval_df = load_data(args.data_path, args.sample_size, args.test_size, args.seed)
    print(f"Train examples: {len(train_df)} | Eval examples: {len(eval_df)}")

    print(f"Loading base model: {args.model_name}")
    model, tokenizer = build_model(args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout)

    train_dataset = SentimentDataset(
        train_df["review"].tolist(), train_df["label"].tolist(), tokenizer, args.max_length
    )
    eval_dataset = SentimentDataset(
        eval_df["review"].tolist(), eval_df["label"].tolist(), tokenizer, args.max_length
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Final evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Model saved to {args.save_dir}")


if __name__ == "__main__":
    main()
