# Job Scam Detection: DistilBERT vs TF-IDF+SVM

A comparative benchmark of classical machine learning vs transformer-based deep learning for detecting fraudulent job postings.

## Overview

This project evaluates two approaches to classifying job postings as **genuine** or **fraudulent**:

1. **Classical baseline:** TF-IDF features + LinearSVC (calibrated)
2. **Deep learning:** Fine-tuned `distilbert-base-uncased` using PyTorch and Hugging Face `transformers`

## Dataset

- Source: [`gplsi/fake_job_postings_balanced_en`](https://huggingface.co/datasets/gplsi/fake_job_postings_balanced_en) (Hugging Face)
- 1,732 real job postings, balanced 50/50 between genuine and fraudulent
- Text fields combined: title, company profile, description, requirements, benefits

## Results

| Model | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---|---|---|
| TF-IDF + LinearSVC | 0.9168 | 0.9164 | **0.9164** |
| Fine-tuned DistilBERT | 0.9225 | 0.9222 | **0.9222** |

Fine-tuning a transformer model improved macro-F1 by ~0.6 points over the classical baseline, demonstrating the benefit of contextual embeddings for nuanced fraud-indicator language in job postings.

## Approach

- **Preprocessing:** Combined relevant text fields into a single input string per posting; train/test split (80/20, stratified)
- **Baseline:** `TfidfVectorizer` (unigrams + bigrams, 8000 features, sublinear TF) → `LinearSVC` with balanced class weights, calibrated for probability outputs
- **Transformer:** `DistilBertTokenizerFast` for tokenization (max length 256) → `DistilBertForSequenceClassification` fine-tuned for 3 epochs via Hugging Face `Trainer` (lr=2e-5, batch size 16)

## Tech Stack

Python, PyTorch, Hugging Face Transformers & Datasets, Scikit-Learn, Pandas, NumPy

## Usage

Open `job_scam_distilbert_vs_svm.ipynb` in Google Colab (GPU runtime recommended) and run cells in order.

## Future Work

- Experiment with larger transformer models (RoBERTa, BERT-base)
- Add explainability (SHAP/LIME) to interpret model decisions
- Deploy as a REST API for real-time scoring
