import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------------
# CONFIG — UPDATE THIS WITH YOUR HF MODEL REPO
# -----------------------------------------------------------
HF_MODEL_NAME = "arti-gupta/media-bias-lora-distilbert"  
# Make sure this repo exists on HuggingFace

# -----------------------------------------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------------------------------------
def load_model():
    """
    Loads DistilBERT + LoRA model from HuggingFace.
    Returns: tokenizer, model (on CPU or GPU)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)

    model.eval()

    return tokenizer, model, device


# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict_text(text: str, tokenizer, model, device):
    """
    Takes input text → returns predicted label.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    # 3-class mapping
    label_map = {
        0: "NEUTRAL 🟦",
        1: "LEFT-LEANING 🔵",
        2: "RIGHT-LEANING 🔴"
    }

    return label_map.get(pred, "UNKNOWN")
