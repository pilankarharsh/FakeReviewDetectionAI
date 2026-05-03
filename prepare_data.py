"""
prepare_data.py
────────────────
Loads the real-labeled Mexwell / OSF fake reviews dataset (kaggle.csv),
cleans text, and splits into train / val / test.

Dataset: 40,432 rows | CG = fake | OR = genuine | perfectly balanced
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\HarXh\Downloads\ppt"
RAW  = f"{BASE}/data/raw/kaggle.csv"  # Use existing kaggle.csv file
OUT  = f"{BASE}/data/processed"
os.makedirs(OUT, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(RAW)
print(f"Loaded: {df.shape[0]} rows | Columns: {df.columns.tolist()}")
print(f"Label counts:\n{df['label'].value_counts()}\n")

# ── Rename & map labels ───────────────────────────────────────────────────────
df = df.rename(columns={"text_": "text"})
df["fake_label"] = df["label"].map({"CG": 1, "OR": 0})   # CG=fake, OR=genuine

# ── Drop nulls ────────────────────────────────────────────────────────────────
before = len(df)
df = df.dropna(subset=["text", "fake_label"]).reset_index(drop=True)
print(f"Dropped {before - len(df)} null rows. Remaining: {len(df)}")

# ── Clean text ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text

print("Cleaning text...")
df["text"] = df["text"].apply(clean_text)

# ── Feature signals (kept for the app's signal display) ──────────────────────
def compute_signals(text):
    words      = text.split()
    wc         = len(words)
    unique     = len(set(w.lower() for w in words))
    lex_div    = round((unique / (wc + 1)) * 100, 1)
    excl       = text.count("!")
    excl_ratio = round((excl / (wc + 1)) * 100, 2)
    avg_wl     = round(sum(len(w) for w in words) / (wc + 1), 2)
    return pd.Series({
        "word_count":        wc,
        "lex_div":           lex_div,
        "excl_ratio":        excl_ratio,
        "avg_word_length":   avg_wl,
    })

print("Computing text signals...")
df[["word_count", "lex_div", "excl_ratio", "avg_word_length"]] = \
    df["text"].apply(compute_signals)

# ── Keep final columns ────────────────────────────────────────────────────────
df = df[["text", "category", "rating", "label", "fake_label",
         "word_count", "lex_div", "excl_ratio", "avg_word_length"]]

# ── Stats ─────────────────────────────────────────────────────────────────────
print(f"\n── Dataset stats ──────────────────────────────")
print(f"Total rows  : {len(df)}")
print(f"Fake (CG)   : {df['fake_label'].sum()} ({df['fake_label'].mean()*100:.1f}%)")
print(f"Genuine (OR): {(df['fake_label']==0).sum()} ({(df['fake_label']==0).mean()*100:.1f}%)")
print(f"Avg words   : {df['word_count'].mean():.0f}")
print(f"───────────────────────────────────────────────\n")

# ── Save full cleaned file ────────────────────────────────────────────────────
df.to_csv(f"{OUT}/full_clean.csv", index=False)
print("Saved: full_clean.csv")

# ── Train / Val / Test split (70 / 15 / 15) ──────────────────────────────────
train, temp = train_test_split(
    df, test_size=0.30,
    stratify=df["fake_label"], random_state=42
)
val, test = train_test_split(
    temp, test_size=0.50,
    stratify=temp["fake_label"], random_state=42
)

train.to_csv(f"{OUT}/train.csv", index=False)
val.to_csv(  f"{OUT}/val.csv",   index=False)
test.to_csv( f"{OUT}/test.csv",  index=False)

print(f"\n── Splits saved ────────────────────────────────")
print(f"Train : {len(train):>6} rows | Fake: {train['fake_label'].mean()*100:.1f}%")
print(f"Val   : {len(val):>6} rows | Fake: {val['fake_label'].mean()*100:.1f}%")
print(f"Test  : {len(test):>6} rows | Fake: {test['fake_label'].mean()*100:.1f}%")
print(f"───────────────────────────────────────────────")
print("\n✅ prepare_data.py complete. Run train_bert.py next.")