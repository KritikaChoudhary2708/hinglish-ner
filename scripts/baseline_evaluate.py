from pathlib import Path
from collections import defaultdict
import spacy
from spacy.tokens import DocBin
from gliner import GLiNER

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data/processed/test.spacy"
LABELS = ["PERSON","ORG", "LOCATION", "PRODUCT", "DATE", "MONEY"]
THRESHOLD = 0.35

#Load test

print("Load test set")
nlp = spacy.blank("hi")
db = DocBin().from_disk(TEST_PATH)
test_docs = list(db.get_docs(nlp.vocab))

print(f"Loaded {len(test_docs)} test doc")

# load Gliner
print("\nLoading GLiNER")
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
print("GLiNER loaded")

# evaluate
print("Evaluating")
tp = defaultdict(int) # true positive
fp = defaultdict(int) # false positive
fn = defaultdict(int) # false negative

for doc in test_docs:
          gold = set()
          for e in doc.ents:
                    gold.add((int(e.start_char), int(e.end_char), e.label_))
          preds = model.predict_entities(doc.text, LABELS, threshold=THRESHOLD)
          pred = set()
          for e in preds:
                    pred.add((int(e["start"]), int(e["end"]), e["label"]))
          for p in pred:
                    if p in gold:
                              tp[p[2]] += 1
                    else:
                              fp[p[2]] += 1
          for g in gold:
                    if g not in pred:
                              fn[g[2]] += 1
          
#Reports

print(f"{'Entity':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
all_tp = all_fp = all_fn = 0

for label in LABELS:
    p  = tp[label] / (tp[label] + fp[label] + 1e-9)
    r  = tp[label] / (tp[label] + fn[label] + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    all_tp += tp[label]
    all_fp += fp[label]
    all_fn += fn[label]
    print(f"{label:<12} {p:>10.3f} {r:>8.3f} {f1:>8.3f}")
 
print("-"*50)
p  = all_tp / (all_tp + all_fp + 1e-9)
r  = all_tp / (all_tp + all_fn + 1e-9)
f1 = 2 * p * r / (p + r + 1e-9)
print(f"{'OVERALL':<12} {p:>10.3f} {r:>8.3f} {f1:>8.3f}")