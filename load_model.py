import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODEL_PATH = "model"  # Make sure your model folder is in Colab at /content/model

def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    return tokenizer, model
