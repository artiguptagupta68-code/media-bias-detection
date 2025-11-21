import streamlit as st, os, pickle, re, nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, AutoModel
from peft import PeftModel
import torch
nltk.download('stopwords'); nltk.download('wordnet')

base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=K)
model = PeftModel.from_pretrained(base_model, "lora_adapter")

st.set_page_config(layout="wide", page_title="Media Bias Detection")
st.title("📰 Media Bias Detection (LoRA adapter)")

# label mapping (from training)
label_map = {"0": "left", "1": "neutral", "2": "right"}

# Prefer loading a LoRA adapter locally and base model from HF hub
nlp = None
try:
    base = "distilbert-base-uncased"
    # load base model from hub (will be downloaded on first run)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(base)
    # load LoRA weights (adapter) from local adapter folder
    model = PeftModel.from_pretrained(model, "lora_adapter")
    nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    st.success("Loaded LoRA adapter + base model from Hugging Face hub.")
except Exception as e:
    st.warning("Could not load LoRA adapter + base model: " + str(e))
    nlp = None

def clean_input(text):
    import re, nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lem.lemmatize(w) for w in text.split() if w not in stop]
    return " ".join(tokens)

txt = st.text_area("Enter a headline", height=120)
if st.button("Predict"):
    if not txt or len(txt.strip()) < 3:
        st.warning("Enter a longer headline")
    else:
        cleaned = clean_input(txt)
        if nlp:
            out = nlp(cleaned)[0]
            st.subheader("DistilBERT + LoRA prediction (scores)")
            for item in out:
                lab = item.get("label")
                score = item.get("score",0)
                try:
                    idx = int(lab.split('_')[-1])
                    st.write(f"{label_map.get(str(idx), str(idx))}: {score*100:.2f}%")
                except:
                    st.write(f"{lab}: {score*100:.2f}%")
        else:
            if clf is None:
                st.error("No model available (neither LoRA nor fallback).")
            else:
                emb = sbert.encode([cleaned])
                pred = clf.predict(emb)[0]
                st.subheader("SBERT + LR fallback prediction")
                st.success(f"Predicted: {label_map.get(str(pred), pred)}")
