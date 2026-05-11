import os 
from pathlib import Path
from typing import List
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Model path

MODEL_PATH = Path(os.getenv("NER_MODEL_PATH","models/hinglish-ner-v1/model-best"))

if not MODEL_PATH.exists():
          raise RuntimeError(f"Model not found: {MODEL_PATH}")


nlp = spacy.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# App

app = FastAPI(
          title ="Hinglish NER API",
          description = "Named Entity Recognition for Hinglish Text",
          version = "1.0.0",
)

# Schema
class TextInput(BaseModel):
          text: str

class BatchInput(BaseModel):
          texts :List[str]

class Entity(BaseModel):
          text:str
          label : str
          start : int
          end : int

class NERResponse(BaseModel): #the original text plus a list of entities found in it
          text: str
          entities: List[Entity]

class batchNERResponse(BaseModel):
          results: List[NERResponse]

# Helpers
def doc_to_response(doc)-> NERResponse:
          return NERResponse(
                    text = doc.text,
                    entities=[
                              Entity(
                                        text = ent.text,
                                        label = ent.label_,
                                        start = ent.start_char,
                                        end = ent.end_char
                              )
                              for ent in doc.ents
                    ]
          )

# Routes
@app.get("/")
def root():
          return{
                    "status":"ok",
                    "model": str(MODEL_PATH),
                    "labels": nlp.get_pipe("ner").labels
                    
          }

@app.post("/ner", response_model= NERResponse)
def predict(payload: TextInput):
          if not payload.text.strip():
                    raise HTTPException(status_code=400,detail={"error":"Empty text provided"})
          
          doc = nlp(payload.text)
          return doc_to_response(doc)

@app.post("/ner/batch", response_model = batchNERResponse)
def predict_batch(payload: BatchInput):
          if not payload.texts:
                    raise HTTPException(status_code=400, details="texts list cannot be empty")
          if len(payload.texts) > 100:
                    raise HTTPException(status_code=400, detail="max 100 texts allowed per batch")
          docs = list(nlp.pipe(payload.texts))
          results = [doc_to_response(doc) for doc in docs]
          return batchNERResponse(results=results)

