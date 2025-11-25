import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# ---------------------------
# DEVICE
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# HuggingFace Model ID
# ---------------------------
HF_MODEL_ID = "arti-gupta/media-bias-lora-distilbert"  # public repo

# ---------------------------
# SIMPLE TEXT PREPROCESSING
# ---------------------------
def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ---------------------------
# LOAD MODEL + TOKENIZER
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ---------------------------
# LABEL MAP
# ---------------------------
label_map = {
    0: "NEUTRAL 🟦",
    1: "LEFT-LEANING 🔵",
    2: "RIGHT-LEANING 🔴"
}

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_bias(text, tokenizer, model):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return label_map[pred]

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Media Bias Detection", layout="centered")
st.title("📰 Media Bias Detection (DistilBERT + LoRA)")
st.write("Enter a news headline to classify its political bias:")

# Load model and tokenizer
tokenizer, model = load_model_and_tokenizer()

headline = st.text_input("Enter News Headline:")
if st.button("Predict Bias"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        result = predict_bias(headline, tokenizer, model)
        st.success(f"Predicted Bias: **{result}**")

st.caption("Model: DistilBERT + LoRA • Trained on 4000 synthetic media headlines")
