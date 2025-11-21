
import streamlit as st
import os, pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer as AutoTokenizer2
import torch
import numpy as np

st.set_page_config(layout="wide")
st.title("Media Bias Detection (LoRA adapter)")

label_map = {"0": "left", "1": "neutral", "2": "right"}

# load fallback
clf = None
if os.path.exists("bias_classifier.pkl"):
    with open("bias_classifier.pkl","rb") as f:
        clf = pickle.load(f)

# SBERT-like embedder via transformers
EMB_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
emb_tok = AutoTokenizer2.from_pretrained(EMB_MODEL)
emb_model = AutoModel.from_pretrained(EMB_MODEL)

def embed_text(text):
    inputs = emb_tok([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outs = emb_model(**inputs)
    return outs.last_hidden_state.mean(dim=1).cpu().numpy()

# try to load base model from HF and then apply local LoRA adapter
nlp = None
try:
    base = "distilbert-base-uncased"
    base_tokenizer = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3)
    base_model = PeftModel.from_pretrained(base_model, "lora_adapter")
    nlp = TextClassificationPipeline(model=base_model, tokenizer=base_tokenizer, return_all_scores=True)
    st.info("Loaded base model from HF and applied local LoRA adapter.")
except Exception as e:
    st.warning("Could not load LoRA+base model: " + str(e))

txt = st.text_area("Enter a headline", height=120)
if st.button("Predict"):
    if not txt or len(txt.strip())<3:
        st.warning("Enter longer headline.")
    else:
        if nlp:
            out = nlp(txt)[0]
            st.subheader("DistilBERT+LoRA prediction scores:")
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
                st.error("No model available.")
            else:
                emb = embed_text(txt)
                pred = clf.predict(emb)[0]
                st.subheader("SBERT-like fallback (LogReg):")
                st.success(f"Predicted: {label_map.get(str(pred), pred)}")
