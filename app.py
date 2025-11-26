import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Optional: PEFT LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except:
    PEFT_AVAILABLE = False

st.set_page_config(page_title="News Bias Detection", layout="wide")

MODEL_DIR = "saved_model"
MODEL_NAME = "distilbert-base-uncased"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # If LoRA adapter files exist, load them
    if PEFT_AVAILABLE and os.path.exists(os.path.join(MODEL_DIR, "adapter_model.bin")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    else:
        model = base_model
    
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Prediction function
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    # Map labels (adjust if your dataset mapping differs)
    labels_map = {0: "neutral", 1: "left", 2: "right"}
    return labels_map.get(pred, "unknown")

# ----------------- Streamlit UI -----------------
st.title("📰 Media Bias Detection")

option = st.radio("Select Input Mode", ["Single Text", "Upload CSV"])

if option == "Single Text":
    text = st.text_area("Enter news headline or article text:")
    if st.button("Predict"):
        if text.strip():
            pred = predict_text(text)
            st.success(f"Prediction: **{pred.upper()}**")
        else:
            st.error("Please enter some text.")

elif option == "Upload CSV":
    file = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            if st.button("Run Batch Prediction"):
                preds = [predict_text(t) for t in df["text"].tolist()]
                df["prediction"] = preds
                st.dataframe(df.head())
                st.download_button(
                    "Download Predictions CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv"
                )
