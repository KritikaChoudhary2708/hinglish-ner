"""
Publish Hinglish NER model + dataset to HuggingFace Hub.
Run from project root: python scripts/publish_model.py
"""

from pathlib import Path
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
HF_USERNAME   = "Kritika00"
MODEL_REPO    = f"{HF_USERNAME}/hinglish-ner-xlmr"
DATASET_REPO  = f"{HF_USERNAME}/hinglish-ner-dataset"
MODEL_PATH    = Path("models/hinglish-ner-v1/model-best")
DATASET_PATH  = Path("data")

api = HfApi()

# ── 1. Push model ─────────────────────────────────────────────────────────────
print(f"\n── Pushing model to {MODEL_REPO} ──")
api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=str(MODEL_PATH),
    repo_id=MODEL_REPO,
    repo_type="model",
)
print("Model pushed.")

# ── 2. Push dataset (processed spacy files + annotations) ────────────────────
print(f"\n── Pushing dataset to {DATASET_REPO} ──")
api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

# Upload processed .spacy files
for split in ["train", "dev", "test"]:
    spacy_file = Path(f"data/processed/{split}.spacy")
    if spacy_file.exists():
        api.upload_file(
            path_or_fileobj=str(spacy_file),
            path_in_repo=f"data/{split}.spacy",
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        print(f"{split}.spacy uploaded.")

# Upload raw annotations
for f in ["annotations.json", "annotations_test.json"]:
    ann_file = Path(f"data/annotated/{f}")
    if ann_file.exists():
        api.upload_file(
            path_or_fileobj=str(ann_file),
            path_in_repo=f"annotations/{f}",
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        print(f"{f} uploaded.")

print("\n🎉 Done!")
print(f"Model  → https://huggingface.co/{MODEL_REPO}")
print(f"Dataset→ https://huggingface.co/datasets/{DATASET_REPO}")