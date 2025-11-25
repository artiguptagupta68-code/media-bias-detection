import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import re

# -------------------------
# Text preprocessing
# -------------------------
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# -------------------------
# Load tokenizer + model (no caching)
# -------------------------
def load_model_and_tokenizer(base_model_name, lora_path=None, hf_token=None):
    with st.spinner("Loading model..."):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_auth_token=hf_token)
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, use_auth_token=hf_token)
        if lora_path:
            model = PeftModel.from_pretrained(base_model, lora_path, use_auth_token=hf_token)
        else:
            model = base_model
        model.eval()
    return tokenizer, model

# -------------------------
# Prediction
# -------------------------
LABEL_MAP = {0: "NEUTRAL 🟦", 1: "LEFT 🔵", 2: "RIGHT 🔴"}

def predict(text, tokenizer, model):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    return LABEL_MAP[pred]

# -------------------------
# Streamlit UI
# -------------------------
st.title("📰 Media Bias Detection (DistilBERT + LoRA)")

MODEL_NAME = "arti-gupta/media-bias-distilbert"
LORA_PATH = "arti-gupta/media-bias-lora"
HF_TOKEN = "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"  # if private model

tokenizer, model = load_model_and_tokenizer(MODEL_NAME, LORA_PATH, HF_TOKEN)

headline = st.text_input("Enter News Headline:")

if st.button("Predict"):
    if not headline.strip():
        st.warning("Please enter a headline")
    else:
        result = predict(headline, tokenizer, model)
        st.success(f"Predicted Bias: **{result}**")
