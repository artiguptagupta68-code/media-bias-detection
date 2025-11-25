import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# ---------------------------
# DEVICE
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# PUBLIC HF MODEL
# ---------------------------
HF_MODEL_ID = "arti-gupta/media-bias-lora-distilbert"  # replace with your public HF repo

# ---------------------------
# TEXT PREPROCESSING
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
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ---------------------------
# LABEL MAP
# ---------------------------
label_map = {0: "NEUTRAL 🟦", 1: "LEFT-LEANING 🔵", 2: "RIGHT-LEANING 🔴"}

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_bias(text, tokenizer, model):
    cleaned = preprocess_text(text)
    inputs = tokenizer(cleaned, ret
