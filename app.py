import streamlit as st
import torch
from load_model import load_classifier
from utils import preprocess_text
from transformers import AutoModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "artiguptagupta68-code/media-bias",
    token=os.environ["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"]
)

tokenizer = AutoTokenizer.from_pretrained(
    "artiguptagupta68-code/media-bias",
    token=os.environ["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"]
)

st.title("📰 Media Bias Detection — DistilBERT + LoRA")
st.write("Classifies a news headline into **Left / Right / Neutral**.")

tokenizer, model = load_classifier()

labels = ["neutral", "left", "right"]

headline = st.text_input("Enter headline:")

if st.button("Predict"):
    clean = preprocess_text(headline)
    inputs = tokenizer(clean, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    st.subheader("Prediction:")
    st.success(labels[pred])
