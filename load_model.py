
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

def load_classifier():
    model_path = "trained_lora_model"  # replace with your folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return model, tokenizer, sbert
