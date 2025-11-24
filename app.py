import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------
# Load HF Token (must be set as env variable)
# ------------------------------------
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

if HF_TOKEN is None:
    st.error("❌ ERROR: HuggingFace token not found.\n\nSet environment variable: HUGGINGFACE_HUB_TOKEN")
    st.stop()

# ------------------------------------
# Load model + tokenizer from private repo
# ------------------------------------
MODEL_ID = "artiguptagupta68-code/media-bias"

@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )
    return tokenizer, model

tokenizer, model = load_classifier()

# ------------------------------------
# Streamlit UI
# ------------------------------------
st.title("📰 Media Bias Detection — DistilBERT + LoRA")
st.write("Classifies a news headline into **Left / Right / Neutral**.")

labels = ["neutral", "left", "right"]

headline = st.text_input("Enter a news headline:")

if st.button("Predict"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    else:
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        st.subheader("Prediction:")
        st.success(labels[pred])
