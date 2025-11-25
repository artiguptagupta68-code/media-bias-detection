import streamlit as st
import torch
from transformers import AutoTokenizer
from peft import PeftModel, LoraConfig, PeftConfig
from transformers import AutoModelForSequenceClassification
import re


def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

@st.cache_resource
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
    
def load_model(model_name, lora_path=None):
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if lora_path:
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        model = base_model
    model.eval()
    return model
LABEL_MAP = {0: "NEUTRAL 🟦", 1: "LEFT 🔵", 2: "RIGHT 🔴"}

def predict(text, tokenizer, model):
    cleaned = preprocess_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return LABEL_MAP[pred]

def predict_bias(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    label_map = {
        0: "🟦 Left Bias",
        1: "⚪ Neutral",
        2: "🟥 Right Bias"
    }

    return label_map[pred]


# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Media Bias Detector", layout="centered")
st.title("📰 Media Bias Detection (DistilBERT + LoRA)")

st.write("Enter a news headline below to detect political bias.")


tokenizer = load_tokenizer(MODEL_NAME)  # cache safe
model = load_model(MODEL_NAME, LORA_PATH)

headline = st.text_input("Enter News Headline:")

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_bias(headline, tokenizer, model)
        st.success(f"Prediction: **{result}**")
