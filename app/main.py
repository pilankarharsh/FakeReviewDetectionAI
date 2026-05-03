from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "bert")
MAX_LEN    = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at startup ───────────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
print(f"Using device: {DEVICE}")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Fake Review Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Schemas ──────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str
    user_id: str = "unknown"
    product_id: str = "unknown"


class SignalScores(BaseModel):
    word_count: int
    lexical_diversity: float
    exclamation_ratio: float
    avg_word_length: float
    caps_ratio: float
    repetition_score: float


class ReviewResponse(BaseModel):
    verdict: str           # "FAKE" | "GENUINE"
    confidence: float      # 0–100
    fake_probability: float
    genuine_probability: float
    signals: SignalScores
    red_flags: list[str]


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_signals(text: str) -> SignalScores:
    words  = text.strip().split()
    wc     = len(words)
    unique = len(set(w.lower() for w in words))
    lex_div = round((unique / (wc + 1)) * 100, 1)

    excl        = len(re.findall(r"!", text))
    excl_ratio  = round((excl / (wc + 1)) * 100, 2)

    avg_wl = round(sum(len(w) for w in words) / (wc + 1), 2)

    caps_words  = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio  = round((caps_words / (wc + 1)) * 100, 2)

    # Repetition: ratio of unique bigrams to total bigrams
    bigrams = [(words[i].lower(), words[i+1].lower()) for i in range(len(words)-1)]
    rep_score = 0.0
    if bigrams:
        rep_score = round((1 - len(set(bigrams)) / len(bigrams)) * 100, 1)

    return SignalScores(
        word_count=wc,
        lexical_diversity=lex_div,
        exclamation_ratio=excl_ratio,
        avg_word_length=avg_wl,
        caps_ratio=caps_ratio,
        repetition_score=rep_score,
    )


def detect_red_flags(text: str, signals: SignalScores, fake_prob: float) -> list[str]:
    flags = []
    if signals.exclamation_ratio > 5:
        flags.append("Excessive exclamation marks")
    if signals.caps_ratio > 10:
        flags.append("Unusual ALL-CAPS usage")
    if signals.lexical_diversity < 40:
        flags.append("Low lexical diversity")
    if signals.repetition_score > 30:
        flags.append("High phrase repetition")
    if signals.word_count < 15:
        flags.append("Very short review")
    if re.search(r'\b(best|amazing|perfect|excellent|worst|terrible)\b.*\1', text.lower()):
        flags.append("Repeated superlatives")
    generic = ["highly recommend", "five stars", "must buy", "everyone should"]
    for phrase in generic:
        if phrase in text.lower():
            flags.append(f'Generic phrase: "{phrase}"')
    return flags


def predict(text: str):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # probs[0] = genuine (label 0), probs[1] = fake (label 1)
    fake_prob    = float(probs[1])
    genuine_prob = float(probs[0])
    verdict      = "FAKE" if fake_prob > 0.5 else "GENUINE"
    confidence   = round(max(fake_prob, genuine_prob) * 100, 1)

    return verdict, confidence, round(fake_prob * 100, 1), round(genuine_prob * 100, 1)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/predict", response_model=ReviewResponse)
def predict_review(req: ReviewRequest):
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Review text too short.")

    verdict, confidence, fake_prob, genuine_prob = predict(req.text)
    signals   = compute_signals(req.text)
    red_flags = detect_red_flags(req.text, signals, fake_prob / 100)

    return ReviewResponse(
        verdict=verdict,
        confidence=confidence,
        fake_probability=fake_prob,
        genuine_probability=genuine_prob,
        signals=signals,
        red_flags=red_flags,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_path": MODEL_PATH,
    }