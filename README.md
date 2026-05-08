# Hinglish NER (Named Entity Recognition)

This repository contains a project for performing Named Entity Recognition (NER) on Hinglish (Hindi + English) text. It includes scripts for gathering data, setting up zero-shot baselines using GLiNER, converting annotated data to spaCy format, and configuring the training pipeline for custom NER models using `xlm-roberta-base`.

## Project Structure

- **`data/`**: Contains all datasets.
  - `raw/`: Raw downloaded texts (e.g., YouTube comments).
  - `annotated/`: Manual annotations stored in JSON format (`annotations.json`).
  - `processed/`: Datasets converted into SpaCy's binary format (`train.spacy`, `dev.spacy`, `test.spacy`) for training and evaluation.
- **`scripts/`**: 
  - `download_youtube_comments.py`: Downloads, deduplicates, and shuffles comments from YouTube videos.
  - `baseline_gliner.py`: Script for zero-shot NER inference using the GLiNER model.
  - `baseline_evaluate.py`: Evaluates the GLiNER model's performance on the test set (`test.spacy`) and calculates Precision, Recall, and F1 scores.
  - `convert_to_spacy.py`: Splits the annotated JSON data into train, validation, and test sets, and converts them to `.spacy` format.
- **`config.cfg`**: The SpaCy configuration file used for training the NER pipeline. It relies on the `xlm-roberta-base` transformer model.

## Setup

1. Clone the repository.
2. Install the necessary Python packages:
   ```bash
   pip install spacy spacy-transformers gliner google-api-python-client python-dotenv
   ```
3. To download YouTube comments, you will need a YouTube Data API Key. Create a `.env` file in the root directory and add your key:
   ```env
   YOUTUBE_API_KEY=your_api_key_here
   ```

## Usage

### 1. Data Collection
Download raw comments using the YouTube API:
```bash
python scripts/download_youtube_comments.py
```

### 2. Data Preparation
Convert the annotated JSON data into the required SpaCy format. This script also splits the data into train (80%), dev (10%), and test (10%) sets:
```bash
python scripts/convert_to_spacy.py
```

### 3. Baseline Evaluation
Evaluate the GLiNER zero-shot model on the `test.spacy` dataset:
```bash
python scripts/baseline_evaluate.py
```

### 4. Training the SpaCy Model
Train your custom Hinglish NER model using the provided `config.cfg`:
```bash
python -m spacy train config.cfg --output ./output --paths.train ./data/processed/train.spacy --paths.dev ./data/processed/dev.spacy
```
