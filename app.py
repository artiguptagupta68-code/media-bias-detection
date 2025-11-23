
import streamlit as st
from load_model import load_classifier
from utils.preprocess_text import clean_text
import torch

st.title("📰 Media Bias Detection (DistilBERT + LoRA + SBERT)")

model, tokenizer, sbert = load_classifier()

headline = st.text_input("Enter News Headline:")

if st.button("Predict Bias"):
    if headline.strip():
        cleaned = clean_text(headline)
        inputs = tokenizer(cleaned, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        label_map = {0: "Neutral", 1: "Left-Leaning", 2: "Right-Leaning"}
        st.subheader(f"🟩 Prediction: {label_map[pred]}")
    else:
        st.error("Please enter a headline.")
