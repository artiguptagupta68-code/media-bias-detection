import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

MODEL_PATH = "./distilbert_lora_media_bias"

def preprocess_text(text):
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

label_map = {0:"NEUTRAL",1:"LEFT-LEANING",2:"RIGHT-LEANING"}

def predict(text, tokenizer, model):
    cleaned = preprocess_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inputs)
        pred = out.logits.argmax(-1).item()
    return label_map[pred]
