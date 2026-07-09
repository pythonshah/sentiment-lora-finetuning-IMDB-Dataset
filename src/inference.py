"""
inference.py
------------
Run sentiment predictions using a fine-tuned LoRA model.

Usage (single sentence):
    python src/inference.py --model_dir ./my_first_finetuned_model --text "This movie was fantastic!"

Usage (interactive mode):
    python src/inference.py --model_dir ./my_first_finetuned_model
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = {0: "negative", 1: "positive"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned sentiment model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the saved fine-tuned model directory")
    parser.add_argument("--text", type=str, default=None,
                        help="A single review text to classify. If omitted, runs interactive mode.")
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(text: str, model, tokenizer, max_length: int = 256):
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred_id = int(torch.argmax(probs))
    return LABELS[pred_id], float(probs[pred_id])


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model_dir)

    if args.text:
        label, confidence = predict(args.text, model, tokenizer, args.max_length)
        print(f"Text: {args.text}")
        print(f"Prediction: {label} (confidence: {confidence:.2%})")
        return

    print("Interactive mode. Type a review and press Enter (type 'exit' to quit).")
    while True:
        text = input("\nReview: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        label, confidence = predict(text, model, tokenizer, args.max_length)
        print(f"Prediction: {label} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
