import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# ---------------------------
# 1) TEXT PREPROCESSOR
# ---------------------------
def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ---------------------------
# 2) LOAD MODEL + LORA
# ---------------------------
HF_TOKEN = "None"  # if private, else None
MODEL_NAME = "arti-gupta/media-bias-lora-distilbert"

@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        use_auth_token=HF_TOKEN
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# ---------------------------
# 3) PREDICTION
# ---------------------------
label_map = {0: "NEUTRAL 🟦", 1: "LEFT-LEANING 🔵", 2: "RIGHT-LEANING 🔴"}

def predict(text, tokenizer, model, device):
    cleaned = preprocess_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    return label_map[pred]

# ---------------------------
# 4) STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Media Bias Detection", layout="centered")
st.title("📰 Media Bias Detection (DistilBERT + LoRA)")
st.write("Enter a news headline to classify its political bias.")

tokenizer, model, device = load_classifier()

headline = st.text_input("Enter News Headline:")

if st.button("Predict Bias"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        result = predict(headline, tokenizer, model, device)
        st.success(f"Predicted Bias: **{result}**")

st.markdown("---")
st.caption("Model: DistilBERT + LoRA • Trained on 4000 synthetic media headlines")
