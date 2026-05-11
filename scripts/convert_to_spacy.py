import json
import random
from collections import Counter
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT/ "data/annotated/annotations.json"
INPUT_TEST = ROOT / "data/annotated/annotations_test.json"
OUT_DIR = ROOT/"data/processed"
SPLIT = (0.8,0.1,0.1)
SEED=42

# Load annotations
print(f"Loading annotations from {INPUT}")
with open(INPUT, encoding = "utf-8") as f:
          raw = json.load(f)
with open(INPUT_TEST, encoding = "utf-8") as f:
          raw_test = json.load(f)

#filtering label and non label entitites

labeled = [t for t in raw if "label" in t and t['label']]
unlabeled = [t for t in raw if "label" not in t or not t['label']]
print(f"Found {len(labeled)} labeled tasks and {len(unlabeled)} unlabeled tasks. Total = {len(raw)}")

all_tasks = labeled + unlabeled
random.seed(SEED)
random.shuffle(all_tasks)
random.seed(SEED)
random.shuffle(raw_test)
#split

n = len(all_tasks)
n_train = int(n*SPLIT[0])
n_val = int(n*SPLIT[1])

train_set = all_tasks[:n_train]
val_set = all_tasks[n_train : n_train + n_val]
test_set = raw_test

print(f"Split: Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")

# Convert to spacy docbin

nlp = spacy.blank("xx")

def make_docbin(tasks):
          db = DocBin()
          skipped = 0
          for task in tasks:
                    text = task["text"]
                    doc = nlp.make_doc(text)
                    spans = []
                    for ent in task.get("label",[]):
                              start_char = ent["start"]
                              end_char = ent["end"]
                              label = ent["labels"][0]
                              span = doc.char_span(start_char, end_char, label=label, alignment_mode="contract")
                              if span == None:
                                        skipped+=1
                                        continue
                              spans.append(span)
                    doc.ents = filter_spans(spans)
                    db.add(doc)
          return db, skipped

print(f"\n Converting to DocBin")
OUT_DIR.mkdir(parents=True, exist_ok=True)
train_db, s1 = make_docbin(train_set)
val_db, s2 = make_docbin(val_set)
test_db, s3 = make_docbin(test_set)
print(f"Skipped spans: Train={s1}|Val={s2}|Test={s3} ||| Total = {s1+s2+s3}")
train_db.to_disk(OUT_DIR/'train.spacy')
val_db.to_disk(OUT_DIR/'dev.spacy')
test_db.to_disk(OUT_DIR/'test.spacy')
print(f"\nSaved to {OUT_DIR}")


#checking

db_check = DocBin().from_disk(OUT_DIR/'train.spacy')
docs = list(db_check.get_docs(nlp.vocab))
counts = Counter(ent.label_ for doc in docs for ent in doc.ents)
for d in docs[:3]:
          print(d.text)
          print(d.ents)
          print("-"*40)

print("\nTrain set entity distribution:")
for label, count in counts.most_common():
    print(f"  {label:10s}: {count}")
 