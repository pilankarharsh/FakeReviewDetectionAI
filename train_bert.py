import os
import re
import torch
import pandas as pd
import nltk
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── CONFIG ─────────────────────────────────────────────────────────────
BASE = r"C:\Users\HarXh\Downloads\ppt"   # change if needed
os.makedirs(f"{BASE}/models/bert", exist_ok=True)

# ── TEXT CLEANING ───────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(t)
              for t in text.split()
              if t not in stop_words]
    return ' '.join(tokens)

# ── LOAD DATA ───────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(f"{BASE}/data/processed/train.csv")
val_df   = pd.read_csv(f"{BASE}/data/processed/val.csv")

# Sample to run faster — remove these 2 lines for full training
train_df = train_df.sample(20000, random_state=42)
val_df   = val_df.sample(4000,   random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# Clean text if needed
text_col = 'clean_text' if 'clean_text' in train_df.columns else 'text'
print(f"Using column: '{text_col}'")

if text_col == 'text':
    print("Cleaning text — takes 2-3 mins...")
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    val_df['clean_text']   = val_df['text'].apply(clean_text)
    print("Cleaning done.")

train_df['clean_text'] = train_df['clean_text'].fillna('').astype(str)
val_df['clean_text']   = val_df['clean_text'].fillna('').astype(str)

# ── DATASET ─────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(
            df['clean_text'].tolist(),
            truncation=True, padding=True, max_length=256)
        self.labels = df['fake_label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

print("Tokenizing...")
train_ds = ReviewDataset(train_df)
val_ds   = ReviewDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

# ── MODEL ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# ── TRAINING ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(3):
    # Training
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/3 Train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/3 Val  "):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total   += len(batch['labels'])

    val_acc = correct / total
    print(f"Epoch {epoch+1} | Val Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(f"{BASE}/models/bert")
        tokenizer.save_pretrained(f"{BASE}/models/bert")
        print(f"  ✓ Best model saved (acc={val_acc:.4f})")

print(f"\nDone. Best Val Accuracy: {best_val_acc:.4f}")
print(f"Model saved to: {BASE}/models/bert")