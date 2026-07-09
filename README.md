# Sentiment Classification with LoRA Fine-Tuning (DistilBERT)

[![CI](https://github.com/<your-username>/sentiment-lora-finetuning/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/sentiment-lora-finetuning/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Parameter-efficient fine-tuning of **DistilBERT** for binary sentiment classification (positive/negative) on the full 50,000-review IMDB Movie Reviews dataset, using **LoRA (Low-Rank Adaptation)** via HuggingFace's PEFT library — with a full experimental workflow around it: a classical baseline for comparison, a rank ablation study, model explainability, out-of-distribution robustness testing, and a live interactive demo.

Instead of updating all ~67M parameters of DistilBERT, this project injects small trainable low-rank adapter matrices into the attention layers (`q_lin`, `v_lin`) and trains only those — reducing trainable parameters by over 99% while retaining strong downstream performance.

## Features

- 🔧 **LoRA fine-tuning** on the full 50k-review dataset — trains a small fraction of total model parameters
- 📊 **Classical ML baseline** (TF-IDF + Logistic Regression) to quantify what the fine-tuned transformer actually gains
- 🧪 **Stratified train/val/test split** with early stopping on validation F1
- 📈 **Full evaluation suite** — accuracy, F1, precision, recall, confusion matrix, classification report, training curves
- 🔬 **LoRA rank ablation study** (r=4/8/16) — trainable-parameter count vs. performance trade-off
- 🕵️ **Explainability with LIME** — visualizes which words drove each individual prediction
- 🌍 **Out-of-distribution robustness testing** — sarcasm, negation, and non-movie domains, to honestly surface failure modes
- 🎛️ **Live Gradio demo** (`app.py`) — interactive web app, shareable link, deployable to Hugging Face Spaces
- 🖥️ **CLI-driven training/inference** — configurable via command-line arguments, no notebook required for production use
- ✅ **Unit tests + CI** — GitHub Actions runs tests and linting on every push
- 📄 **Model card** — documents intended use, training data, and limitations (`MODEL_CARD.md`)
- 🧩 **Modular code** — dataset, training, and inference cleanly separated

## Project Structure

```
sentiment-lora-finetuning/
├── src/
│   ├── dataset.py       # PyTorch Dataset wrapper for tokenized reviews
│   ├── train.py         # Training entry point (LoRA + Trainer, CLI-driven)
│   └── inference.py     # Load a fine-tuned model and run predictions
├── notebooks/
│   └── sentiment_lora_finetuning.ipynb   # Full workflow: EDA, baseline, training,
│                                          # evaluation, ablation, LIME, OOD test, Gradio demo
├── tests/
│   ├── test_dataset.py       # Unit tests for the Dataset class (no model download needed)
│   └── test_data_loading.py  # Unit tests for CSV loading / label mapping / splitting
├── .github/workflows/ci.yml  # GitHub Actions: lint + unit tests on every push
├── app.py                # Standalone Gradio demo app
├── MODEL_CARD.md          # Model card: intended use, training data, limitations
├── requirements.txt
├── requirements-dev.txt   # Adds testing, linting, and notebook/demo dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/sentiment-lora-finetuning.git
cd sentiment-lora-finetuning
pip install -r requirements-dev.txt   # includes testing, linting, notebook + demo extras
```

> **Note:** If running on Google Colab, ensure `torchao` is not installed alongside older `peft` versions — it can cause an `ImportError` on `get_peft_model()`. Uninstall it if present: `pip uninstall -y torchao`.

## Dataset

This project expects the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (or any CSV with `review` and `sentiment` columns, where `sentiment` is `positive`/`negative`).

Place the CSV anywhere and pass its path via `--data_path`. The training script and notebook use the **full 50,000-row dataset** by default (not a small sample).

## Full Workflow (Recommended: run the notebook first)

`notebooks/sentiment_lora_finetuning.ipynb` is the primary artifact of this project and walks
through the entire pipeline end-to-end — open it in Google Colab or Jupyter:

1. Exploratory data analysis (class balance, review length, word clouds)
2. TF-IDF + Logistic Regression baseline
3. LoRA fine-tuning of DistilBERT on the full dataset, with early stopping
4. Evaluation: confusion matrix, classification report, training curves
5. Error analysis on misclassified examples
6. LoRA rank ablation study (r=4/8/16)
7. LIME explainability on individual predictions
8. Out-of-distribution robustness testing
9. Live Gradio demo with a shareable link

The `src/` scripts below are the production-oriented, CLI-driven equivalents of steps 3–4,
for anyone who wants to train/run inference outside of a notebook.

## Training

```bash
python src/train.py \
    --data_path data/IMDB_Dataset.csv \
    --sample_size 1200 \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-4
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_path` | *(required)* | Path to the IMDB CSV file |
| `--sample_size` | `1200` | Rows sampled from the full dataset (`-1` = use all) |
| `--epochs` | `3` | Training epochs |
| `--batch_size` | `16` | Per-device batch size |
| `--learning_rate` | `2e-4` | LoRA learning rate |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA scaling factor |

The fine-tuned adapter weights are saved to `./my_first_finetuned_model` by default.

## Inference

Single prediction:

```bash
python src/inference.py --model_dir ./my_first_finetuned_model --text "This movie was absolutely brilliant!"
```

Interactive mode:

```bash
python src/inference.py --model_dir ./my_first_finetuned_model
```

## Live Demo

```bash
python app.py --model_dir ./my_first_finetuned_model
```

Launches a local Gradio web app for interacting with the model. The same interface is also
included at the end of the notebook with `share=True`, which produces a temporary public
link — handy for including in a portfolio or resume.

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

`tests/test_dataset.py` runs with no external dependencies beyond `torch` (uses a fake
tokenizer). `tests/test_data_loading.py` exercises the CSV loading and splitting logic in
`train.py` against a small synthetic CSV. CI runs the dataset tests plus linting on every
push (see `.github/workflows/ci.yml`).

## Results

*(Fill in with your actual metrics after training — see `test_metrics.json` and
`ablation_results.json` generated by the notebook)*

| Metric | LoRA Fine-Tuned DistilBERT | TF-IDF + Logistic Regression Baseline |
|---|---|---|
| Accuracy | 0.XX | 0.XX |
| F1 Score | 0.XX | 0.XX |
| Precision | 0.XX | — |
| Recall | 0.XX | — |

### LoRA Rank Ablation

| Rank (r) | Trainable Params | Trainable % | Validation F1 |
|---|---|---|---|
| 4 | ... | ... | ... |
| 8 | ... | ... | ... |
| 16 | ... | ... | ... |

See `MODEL_CARD.md` for a full write-up of intended use, training data, and known limitations.

## Why LoRA?

Full fine-tuning of transformer models requires updating every parameter, which is memory- and compute-intensive. LoRA freezes the pretrained weights and injects small trainable rank-decomposition matrices into selected layers. This project targets the query and value projection layers (`q_lin`, `v_lin`) of DistilBERT's attention blocks, drastically cutting the number of trainable parameters while preserving most of full fine-tuning's performance — making it practical to train on a single free-tier GPU (e.g., Google Colab).

## License

This project is released under the [MIT License](LICENSE).

## Author

Built by **Adnan** — freelance developer & designer.
