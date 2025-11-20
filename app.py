
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import pickle

st.title("📰 Media Bias Classifier")

# Load model
model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

# Load Logistic Regression fallback
with open("bias_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

user_text = st.text_area("Enter news text:", "")

if st.button("Predict Bias"):
    if len(user_text) < 5:
        st.warning("Please enter more text.")
    else:
        result = nlp(user_text)[0]['label']
        st.success(f"Predicted Bias: {result}")
