import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

@st.cache_resource
def load_model():
    base_model_id = "distilbert-base-uncased"
    lora_path = "./distilbert_lora_media_bias"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # MUST match training labels count
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=3  # Left, Neutral, Right
    )

    # Load LoRA adapter on top of base model
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    return tokenizer, model


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

tokenizer, model = load_model()

headline = st.text_input("Enter News Headline:")

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_bias(headline, tokenizer, model)
        st.success(f"Prediction: **{result}**")
