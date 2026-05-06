import json
import time
from pathlib import Path
from gliner import GLiNER
from collections import Counter
import random
# config
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data/raw/hinglish_raw_shuffled.jsonl"
OUT_PATH = ROOT / "data/annotated/gliner_preannotared.json"
MODEL_NAME = "urchade/gliner_multi-v2.1"
LABELS=['PERSON','ORG', 'LOCATION', 'PRODUCT', 'DATE', 'MONEY'] 
THRESHOLD= 0.35 
BATCH_SIZE=16

#Load data

print(f"Loading data from {RAW_DATA}...")
sentences = []
with open(RAW_DATA) as f:
          for line in f:
                    obj = json.loads(line)
                    text = obj.get("text","").strip()
                    if text:
                              sentences.append(text)
random.seed(42)
random.shuffle(sentences)
print(f"Loaded {len(sentences)} sentences.")

#Load GLiner (Generalized Language INdependent Entity Recognition)

print(f"\n Loading Gliner model: {MODEL_NAME} ...")
model = GLiNER.from_pretrained(MODEL_NAME)
print("Model ready")

print(f"\n Running Gliner on {len(sentences)} sentences (batch={BATCH_SIZE})")
ls_tasks = []
total_entities = 0
start = time.time()
for i in range(0, len(sentences), BATCH_SIZE):
          batch = sentences[i: i+BATCH_SIZE]
          try:
                    batch_entities = model.batch_predict_entities(batch, LABELS, THRESHOLD)
          except Exception as e:
                    print(f"Error processing batch {i} to {i+BATCH_SIZE}: {e}")
                    batch_entities = [[] for _ in batch]
          for text, entities in zip(batch,batch_entities):
                    result = []
                    for ent in entities:
                              result.append({
                                        "from_name": 'label',
                                        'to_name' : 'text',
                                        'type' : 'labels',
                                        'value' :{
                                                  'start': ent['start'],
                                                  'end': ent['end'],
                                                  'text': ent['text'],
                                                  'labels': [ent['label']]
                                        }
                              })
                    total_entities += len(result)

                    ls_tasks.append({
                              "data":{"text": text},
                              "annotations":[],
                              "predictions":[
                                        {
                                                  "model_version": MODEL_NAME,
                                                  "score": 1.0,
                                                  "result": result
                                        }
                              ]
                    })
          if (i // BATCH_SIZE) % 10 == 0:
                    done = min(i + BATCH_SIZE, len(sentences))
                    print(f"  {done}/{len(sentences)} sentences processed ...")
 
elapsed = time.time() - start
print(f"\nDone in {elapsed:.1f}s — {total_entities} entity spans found across {len(ls_tasks)} sentences.")
                    
# Output

OUT_PATH.parent.mkdir(parents = True, exist_ok=True)
with open(OUT_PATH, "w", encoding ="utf-8") as f:
          json.dump(ls_tasks, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(ls_tasks)} annotated tasks to {OUT_PATH}")
print("\nNext step: Import this file into Label Studio")
print("  1. Open http://localhost:8080")
print("  2. Create project → NER template (label names: PERSON ORG LOCATION PRODUCT DATE MONEY)")
print("  3. Import → Upload → select gliner_preannotated.json")
print("  4. Gliner's guesses appear as pre-fills — just fix mistakes & submit.")

label_counts = Counter()
for task in ls_tasks:
          for span in task["predictions"][0]["result"]:
                    label_counts[span["value"]["labels"][0]] += 1

print("\n Pre-annotation label distribution:")
for label, count in sorted(label_counts.items()):
          print(f"  {label}: {count}")