"""
app.py
------
Gradio web demo for the LoRA-fine-tuned sentiment classifier.

Run locally:
    python app.py --model_dir ./my_first_finetuned_model

Run on Hugging Face Spaces:
    Set MODEL_DIR env var or place the model files alongside this script,
    then this file works as-is as a Spaces `app.py` entry point.
"""

import argparse
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = {0: "negative", 1: "positive"}


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer


def build_predict_fn(model, tokenizer, max_length: int = 256):
    device = next(model.parameters()).device

    def predict(text: str):
        if not text or not text.strip():
            return {"positive": 0.5, "negative": 0.5}
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()
        return {"positive": float(probs[1]), "negative": float(probs[0])}

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("MODEL_DIR", "./my_first_finetuned_model"),
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    predict_fn = build_predict_fn(model, tokenizer)

    demo = gr.Interface(
        fn=predict_fn,
        inputs=gr.Textbox(lines=4, placeholder="Type a movie review here...", label="Review Text"),
        outputs=gr.Label(num_top_classes=2, label="Predicted Sentiment"),
        title="IMDB Sentiment Classifier (LoRA-tuned DistilBERT)",
        description=(
            "A DistilBERT model fine-tuned with LoRA on 50,000 IMDB reviews. "
            "Enter any movie review to see the predicted sentiment and confidence."
        ),
        examples=[
            ["This film changed the way I see cinema. Absolutely stunning."],
            ["I fell asleep twice. Painfully slow and pointless."],
            ["Not the best, not the worst — a solidly average watch."],
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
