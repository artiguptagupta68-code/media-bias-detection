
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from sentence_transformers import SentenceTransformer
import pickle, os

st.title("📰 Media Bias Detector")

label_map = {'left': 0, 'neutral': 1, 'right': 2}

# Load fallback model
with open("bias_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

sbert = SentenceTransformer("all-MiniLM-L6-v2")

pipeline = None
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert_bias_model", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert_bias_model", local_files_only=True)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
except:
    st.warning("DistilBERT not loaded; using SBERT fallback.")

headline = st.text_area("Enter a news headline:")

def clean(text):
    import re, nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = " ".join([lem.lemmatize(w) for w in text.split() if w not in stop])
    return text

if st.button("Predict"):
    cleaned = clean(headline)

    if pipeline:
        scores = pipeline(cleaned)[0]
        st.subheader("DistilBERT Prediction:")
        for s in scores:
            idx = int(s["label"].split("_")[-1])
            st.write(f"{label_map[idx]}: {s['score']*100:.2f}%")
    else:
        emb = sbert.encode([cleaned])
        pred = clf.predict(emb)[0]
        st.subheader("SBERT Fallback Prediction:")
        st.success(f"Bias: {label_map[pred]}")
