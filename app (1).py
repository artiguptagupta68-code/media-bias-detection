
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

st.set_page_config(page_title="Media Bias Detection", layout="wide")
st.title("Media Bias Detection in News Headlines")

headline = st.text_area("Enter a news headline:")

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("distilbert_bias_model")
    tokenizer = AutoTokenizer.from_pretrained("distilbert_bias_model")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

nlp_pipe = load_model()

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        result = nlp_pipe(headline)
        st.write("Prediction Scores:")
        st.json(result)
