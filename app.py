import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# 1) CONFIGURATION
# -----------------------------
HF_MODEL_ID = "arti-gupta/media-bias-lora-distilbert"
HF_TOKEN = "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"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2) TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# 3) LOAD MODEL + TOKENIZER
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_auth_token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_ID,
        use_auth_token=HF_TOKEN
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# -----------------------------
# 4) LABEL MAP & PREDICTION
# -----------------------------
LABEL_MAP = {0: "NEUTRAL 🟦", 1: "LEFT-LEANING 🔵", 2: "RIGHT-LEANING 🔴"}

def predict(text, tokenizer, model):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=-1).item()
    return LABEL_MAP.get(pred_class, "UNKNOWN")

# -----------------------------
# 5) STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Media Bias Detection", layout="centered")
st.title("📰 Media Bias Detection (DistilBERT + LoRA)")
st.write("Classify a news headline as Neutral, Left-Leaning, or Right-Leaning.")

headline = st.text_input("Enter News Headline:")

if st.button("Predict Bias"):
    if not headline.strip():
        st.warning("Please enter a news headline.")
    else:
        tokenizer, model = load_model_and_tokenizer()
        result = predict(headline, tokenizer, model)
        st.success(f"Predicted Bias: **{result}**")

st.markdown("---")
st.caption("Model: DistilBERT + LoRA • Trained on 4000 synthetic media headlines")
