
import streamlit as st
import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from peft import PeftModel

st.set_page_config(page_title="Media Bias Detection", layout="wide")

BASE_MODEL = "distilbert-base-uncased"
LORA_DIR = "lora_adapter"
LABELS = None

# Load label map
if os.path.exists("label_map.json"):
    with open("label_map.json","r") as f:
        LABELS = json.load(f)
else:
    LABELS = {"0":"left","1":"neutral","2":"right"}

@st.cache_resource
def load_pipeline():
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
        # apply local LoRA adapter
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
        model.eval()
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
        return pipe
    except Exception as e:
        st.error("Could not load LoRA adapter + base model: " + str(e))
        return None

pipe = load_pipeline()

def clean_input(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+","", text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = " ".join(text.split())
    return text

st.title("📰 Media Bias Detection (DistilBERT + LoRA)")

text = st.text_area("Enter a news headline", height=120)

if st.button("Predict"):
    if not text or len(text.strip())<3:
        st.warning("Please enter a longer headline.")
    else:
        cleaned = clean_input(text)
        if pipe is not None:
            out = pipe(cleaned)[0]  # list of dicts
            # map labels
            results = {}
            for item in out:
                lab = item.get("label")
                score = item.get("score",0)
                try:
                    idx = int(lab.split("_")[-1])
                    label_name = LABELS.get(str(idx), str(idx))
                except:
                    label_name = lab
                results[label_name] = round(score, 4)
            st.write(results)
        else:
            st.error("Model pipeline not available.")
