import pandas as pd
import numpy as np
import re
import os
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Reproducibility
import torch
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ====== CONFIG ======
DATA_PATH = "data/imdb/IMDB_Dataset.csv"
SAVE_DIR = "data/imdb/processed/"
VOCAB_SIZE = 10000
SEQ_LENGTHS = [25, 50, 100]
TEST_SIZE = 0.5  # 25k train / 25k test split
MAX_SAMPLES = 50000  # Ensure consistent limit

os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 1. Load Dataset ======
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df = df.head(MAX_SAMPLES)

print(f"Dataset shape: {df.shape}")
print(df.head())

# ====== 2. Basic Preprocessing ======
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("🧹 Cleaning text...")
df["review"] = df["review"].astype(str).apply(clean_text)

# ====== 3. Label Encoding ======
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
labels = df["sentiment"].values

# ====== 4. Split Dataset ======
print("✂️ Splitting dataset 50/50...")
X_train, X_test, y_train, y_test = train_test_split(
    df["review"].values, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ====== 5. Tokenization ======
print("🔢 Tokenizing text...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(f"Vocab size (actual): {len(word_index)}")

# Save tokenizer for reuse in training/evaluation
import pickle
with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# ====== 6. Convert to sequences and pad ======
for seq_len in SEQ_LENGTHS:
    print(f"📏 Processing sequence length = {seq_len}...")

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=seq_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=seq_len, padding='post', truncating='post')

    # Save processed arrays
    np.save(os.path.join(SAVE_DIR, f"X_train_{seq_len}.npy"), X_train_pad)
    np.save(os.path.join(SAVE_DIR, f"X_test_{seq_len}.npy"), X_test_pad)
    np.save(os.path.join(SAVE_DIR, f"y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, f"y_test.npy"), y_test)

print("✅ Preprocessing complete! Files saved in:", SAVE_DIR)
