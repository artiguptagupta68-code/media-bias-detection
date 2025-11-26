# app.py — Streamlit frontend for the DistilBERT+LoRA adapter model
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import joblib
import torch
import pandas as pd

st.set_page_config(page_title="Media Bias Detection", layout="wide")
st.title("📰 Media Bias Detection — Left | Neutral | Right")

MODEL_DIR = "saved_model"   # directory created by training script

@st.cache_resource
def load_model_and_tokenizer(model_dir=MODEL_DIR, base_model_name="distilbert-base-uncased"):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # load base architecture and then apply adapter
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)
    # apply the LoRA adapter weights saved in model_dir
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    # load label encoder
    le = joblib.load(f"{model_dir}/label_encoder.pkl")
    return tokenizer, model, le

tokenizer, model, le = load_model_and_tokenizer()

def predict_text(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        logits = outputs.logits.cpu()
        pred_id = int(torch.argmax(logits, dim=-1).item())
    return le.inverse_transform([pred_id])[0]

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    st.markdown("Load model directory: `saved_model`")
    show_confusion = st.checkbox("Show validation confusion matrix (if available)", value=True)

# Main UI
st.subheader("Predict a single headline")
input_text = st.text_area("Enter a news headline", height=120)
if st.button("Predict single"):
    if input_text.strip() == "":
        st.error("Please enter a headline to predict.")
    else:
        pred = predict_text(input_text)
        st.success(f"Predicted bias: **{pred.upper()}**")

st.markdown("---")
st.subheader("Batch predict from CSV")
uploaded_file = st.file_uploader("Upload CSV file with column 'text' or 'headline' (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Accept both 'text' and 'headline' column names
    if "text" not in df.columns and "headline" not in df.columns:
        st.error("CSV must contain a 'text' or 'headline' column.")
    else:
        col = "text" if "text" in df.columns else "headline"
        if st.button("Run batch prediction"):
            preds = []
            for t in df[col].astype(str).tolist():
                preds.append(predict_text(t))
            df["prediction"] = preds
            st.dataframe(df.head(200))
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")

# Show confusion matrix if exists
if show_confusion:
    try:
        from PIL import Image
        img = Image.open(f"{MODEL_DIR}/confusion_matrix.png")
        st.image(img, caption="Validation confusion matrix", use_column_width=True)
    except Exception:
        st.info("No confusion matrix found in saved_model/ (maybe training not run here).")

st.markdown("---")
st.caption("Model: DistilBERT + LoRA (PEFT). Trained for 30 epochs.")
