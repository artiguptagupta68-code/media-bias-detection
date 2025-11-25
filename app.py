import streamlit as st
from model import load_classifier, predict

st.set_page_config(page_title="Media Bias Detection", layout="centered")
st.title("📰 Media Bias Detection (DistilBERT + LoRA)")

tokenizer, model = load_classifier()

headline = st.text_input("Enter a news headline:")

if st.button("Predict Bias"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        result = predict(headline, tokenizer, model)
        st.success(f"Predicted Bias: **{result}**")
