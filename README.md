# 🇮🇳 Hinglish NER — Named Entity Recognition for Hindi-English Mixed Text

[![Model](https://img.shields.io/badge/🤗%20Model-kritikachoudhary%2Fhinglish--ner--xlmr-blue)](https://huggingface.co/kritikachoudhary/hinglish-ner-xlmr)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-kritikachoudhary%2Fhinglish--ner--dataset-green)](https://huggingface.co/datasets/kritikachoudhary/hinglish-ner-dataset)
[![Demo](https://img.shields.io/badge/🚀%20Demo-Gradio-orange)](http://localhost:7860)

> **The first production-grade open-source NER model for Hinglish (Hindi-English code-mixed) text.**

---

## What is Hinglish?

Hinglish is Hindi-English code-mixed text — how 600M+ Indians actually write online.

```
"Virat ne Mumbai mein Zomato se biryani order ki aur 500 rupaye pay kiye."
→ Virat (PERSON), Mumbai (LOCATION), 500 rupaye (MONEY)
```

No standardised spelling. No existing NER dataset. This project fills that gap.

---

## Model Performance

| Entity      | Precision | Recall | F1    |
|-------------|-----------|--------|-------|
| DATE        | 1.000     | 1.000  | 1.000 |
| ORG         | 0.786     | 0.688  | 0.733 |
| LOCATION    | 0.667     | 0.444  | 0.533 |
| PERSON      | 0.514     | 0.439  | 0.474 |
| PRODUCT     | 0.000     | 0.000  | 0.000 |
| MONEY       | —         | —      | — *(not in fresh test set)* |
| **Overall** | **0.614** | **0.479** | **0.538** |

Evaluated on a **clean held-out test set** (fresh sentences never seen during training).  
Dev F1 during training: **74.10** (step 2,400, P=79.12, R=69.68)  
Baseline (Gliner zero-shot, no fine-tuning): F1 = 59.8

**Known limitations:** PRODUCT generalises poorly (only 115 training examples). PERSON recall is low due to high spelling variation in Hinglish names.

---

## Tech Stack

| Tool | Role |
|------|------|
| spaCy 3.7+ | NER training framework |
| XLM-RoBERTa | Multilingual encoder (pre-trained on 100 languages) |
| Gliner | Zero-shot baseline + pre-annotation |
| Label Studio | Manual annotation |
| FastAPI | REST API deployment |
| Gradio | Interactive demo UI |
| HuggingFace Hub | Model + dataset publishing |
| DVC | Data version control |

---

## Project Structure

```
hinglish-ner/
├── data/
│   ├── raw/                        ← 2,169 collected Hinglish sentences
│   ├── annotated/annotations.json  ← Label Studio export
│   └── processed/                  ← train/dev/test .spacy files
├── models/
│   └── hinglish-ner-v1/model-best/ ← trained spaCy model
├── app/
│   └── main.py                     ← FastAPI server
├── scripts/
│   ├── baseline_gliner.py          ← zero-shot baseline
│   ├── convert_to_spacy.py         ← Label Studio → DocBin
│   └── baseline_evaluate.py        ← evaluation script
├── demo.py                         ← Gradio demo
├── config.cfg                      ← spaCy training config
└── requirements.txt
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/kritikachoudhary/hinglish-ner.git
cd hinglish-ner
python -m venv hinglish-env
source hinglish-env/bin/activate
pip install -r requirements.txt
```

### 2. Download the model

```bash
# From HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download('kritikachoudhary/hinglish-ner-xlmr', local_dir='models/hinglish-ner-v1/model-best')
"
```

### 3. Run FastAPI server

```bash
uvicorn app.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. Run Gradio demo

```bash
python demo.py
# → http://localhost:7860
```

---

## API Reference

### `POST /ner` — Single text

```bash
curl -X POST http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Virat ne Mumbai mein Zomato se biryani order ki"}'
```

**Response:**
```json
{
  "text": "Virat ne Mumbai mein Zomato se biryani order ki",
  "entities": [
    {"text": "Virat",  "label": "PERSON",   "start": 0,  "end": 5},
    {"text": "Mumbai", "label": "LOCATION", "start": 9,  "end": 15}
  ]
}
```

### `POST /ner/batch` — Multiple texts (max 100)

```bash
curl -X POST http://localhost:8000/ner/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Priya Delhi gayi.", "Shah Rukh Khan Netflix pe hai."]}'
```

### `GET /` — Health check

```bash
curl http://localhost:8000/
# {"status": "ok", "model": "...", "labels": ["DATE", "LOCATION", ...]}
```

---

## Entity Types

| Label    | Examples |
|----------|---------|
| PERSON   | Virat, Shah Rukh Khan, Priya |
| ORG      | Google, Zomato, Netflix, Amazon |
| LOCATION | Mumbai, Delhi, Bangalore |
| PRODUCT  | iPhone 15, biryani |
| DATE     | next Friday, kal, agle hafte |
| MONEY    | 500 rupaye, 80000 mein |

---

## Training Details

- **Base model:** XLM-RoBERTa (trained on 100 languages including Hindi + English)
- **Framework:** spaCy 3.7 + spacy-transformers
- **Dataset:** 1,000+ manually annotated Hinglish sentences (YouTube comments, product reviews)
- **Annotation tool:** Label Studio with Gliner pre-annotation
- **Training:** Kaggle T4 GPU, 3,000 steps, LR = 4e-5
- **Best checkpoint:** step 2,400 (F1 = 74.10)

---

## Limitations

- MONEY entity boundaries can be noisy (e.g. includes surrounding words)
- Rare names and novel ORGs may be missed
- Performance degrades on very short texts (< 5 words)
- Not tested on formal Hindi (Devanagari script) — designed for romanised Hinglish only

---

