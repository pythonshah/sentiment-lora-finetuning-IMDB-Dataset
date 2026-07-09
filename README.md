# Sentiment Classification with LoRA Fine-Tuning (DistilBERT)

[![CI](https://github.com/pythonshah/sentiment-lora-finetuning-IMDB-Dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/pythonshah/sentiment-lora-finetuning-IMDB-Dataset/actions/workflows/ci.yml)
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
git clone https://github.com/pythonshah/sentiment-lora-finetuning-IMDB-Dataset.git
cd sentiment-lora-finetuning-IMDB-Dataset
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

Results from the full training run on the complete 50,000-review dataset
(40k train / 5k validation / 5k held-out test, stratified). Raw numbers are saved in
`test_metrics.json` and `ablation_results.json` by the notebook.

| Metric | LoRA Fine-Tuned DistilBERT | TF-IDF + Logistic Regression Baseline |
|---|---|---|
| Accuracy | **0.9124** | 0.9004 |
| F1 Score | **0.9136** | 0.9006 |
| Precision | 0.9015 | — |
| Recall | 0.9260 | — |

**+1.20 accuracy points** over the classical baseline. The gap is modest rather than
dramatic — expected for movie-review sentiment, where strong lexical cues (words like
"brilliant" or "waste of time") already carry most of the signal, leaving less room for a
transformer's contextual understanding to add value compared to harder NLP tasks.

Trainable parameters: **739,586 out of 67,694,596 total (1.09%)** — LoRA updates about 1%
of DistilBERT's weights and still outperforms full-vocabulary TF-IDF features.

Error analysis on the test set: **438 / 5,000 misclassified (8.76%)**. Misclassifications
cluster around reviews with mixed or ambivalent language (e.g. a review criticizing a
"boring" play while still calling elements of the production "clever") — the kind of case
where lexical cues point one way but overall sentiment leans another.

### Out-of-Distribution Robustness Test

A small, hand-picked stress test (9 examples spanning negation, sarcasm, and non-movie
domains) — not a rigorous benchmark, but a useful qualitative signal:

| Category | Result |
|---|---|
| Negation ("not a bad movie") | ✅ Correct — both examples |
| Domain shift (restaurant/product reviews) | ✅ Correct — all 4 examples |
| Sarcasm | ❌ Incorrect — both sarcasm examples misclassified |
| **Overall** | **7 / 9 (77.8%)** |

The clearest failure mode: sarcasm. Both sarcastic examples ("Oh great, another two hours
of my life I'll never get back" and "Wow, truly a 'masterpiece'...") were predicted
positive despite being negative in intent — the model latches onto surface-level positive
words ("great", "masterpiece") without picking up on sarcastic tone. This is a known,
expected limitation of models trained purely on direct-sentiment reviews rather than
sarcasm-labeled data, and is documented in `MODEL_CARD.md`.

### LoRA Rank Ablation

Trained on a fixed 6,000-review subset for 2 epochs per configuration, to isolate the
effect of rank while keeping ablation runtime reasonable (the main model above was trained
on the full dataset):

| Rank (r) | Trainable Params | Trainable % | Validation Accuracy | Validation F1 |
|---|---|---|---|---|
| 4 | 665,858 | 0.985% | 0.8708 | 0.8737 |
| 8 | 739,586 | 1.093% | 0.8783 | 0.8819 |
| 16 | 887,042 | 1.308% | 0.8825 | 0.8858 |

Higher rank gives a small, fairly consistent F1 improvement (~0.012 from r=4 to r=16) at
the cost of ~33% more trainable parameters — a mild, close-to-linear trade-off in this
range. r=8 (used for the main model) sits at a reasonable middle ground between the two.

See `MODEL_CARD.md` for a full write-up of intended use, training data, and known limitations.

## Why LoRA?

Full fine-tuning of transformer models requires updating every parameter, which is memory- and compute-intensive. LoRA freezes the pretrained weights and injects small trainable rank-decomposition matrices into selected layers. This project targets the query and value projection layers (`q_lin`, `v_lin`) of DistilBERT's attention blocks, drastically cutting the number of trainable parameters while preserving most of full fine-tuning's performance — making it practical to train on a single free-tier GPU (e.g., Google Colab).

## License

This project is released under the [MIT License](LICENSE).

## Author

Built by **Adnan** — freelance developer & designer.
