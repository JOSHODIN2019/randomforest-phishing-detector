"""
Run once locally to generate model.pkl, tfidf.pkl, and meta.json.
Commit those three files — the Streamlit app loads them instead of
retraining RandomForest on every cold start.
"""

import os
import re
import json
import pickle
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
for corpus in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4",
               "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(corpus, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Tobi-Dataset.csv")

print("Loading dataset...")
raw = pd.read_csv(DATA_PATH)
raw["email_text"] = raw["subject"].fillna("") + " " + raw["body"].fillna("")
df  = raw[["email_text", "label"]].copy()

total_raw = len(df)
spam_raw  = int((df["label"] == 1).sum())
ham_raw   = int((df["label"] == 0).sum())
print(f"  Loaded {total_raw:,} emails  ({spam_raw:,} spam / {ham_raw:,} ham)")

df = df.drop_duplicates(subset=["email_text"]).reset_index(drop=True)
df = df.dropna(subset=["email_text", "label"]).reset_index(drop=True)

def clean(text):
    t = str(text)
    t = re.sub(r"<[^>]+>",                   " ", t)
    t = re.sub(r"https?://\S+|www\.\S+",     " ", t)
    t = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", t)
    t = re.sub(r"[^\w\s]",                    " ", t)
    t = re.sub(r"\d+",                        " ", t)
    t = re.sub(r"[^a-zA-Z\s]",               " ", t)
    t = re.sub(r"\s+",                        " ", t).strip()
    return t

print("Cleaning text...")
df["email_text"] = df["email_text"].apply(clean).str.lower()

sw = set(stopwords.words("english"))
df["email_text"] = df["email_text"].apply(
    lambda t: " ".join(w for w in t.split() if w not in sw))

all_w = " ".join(df["email_text"]).split()
freq  = Counter(all_w)
frequent_words = set(w for w, _ in freq.most_common(20))
df["email_text"] = df["email_text"].apply(
    lambda t: " ".join(w for w in t.split() if w not in frequent_words))

all_w = " ".join(df["email_text"]).split()
freq  = Counter(all_w)
rare_words = set(w for w, c in freq.items() if c < 2)
df["email_text"] = df["email_text"].apply(
    lambda t: " ".join(w for w in t.split() if w not in rare_words))

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def full_process(text):
    tokens = text.split()
    tokens = [stemmer.stem(w)         for w in tokens]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

print("Preprocessing tokens...")
df["processed"] = df["email_text"].apply(full_process)

print("Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["processed"])
y = df["label"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print("Training RandomForest (100 trees) — this takes a few minutes...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

metrics = {
    "accuracy":  round(accuracy_score (y_te, y_pred) * 100, 2),
    "precision": round(precision_score(y_te, y_pred) * 100, 2),
    "recall":    round(recall_score   (y_te, y_pred) * 100, 2),
    "f1":        round(f1_score       (y_te, y_pred) * 100, 2),
}
print(f"  Accuracy {metrics['accuracy']}%  |  F1 {metrics['f1']}%")

with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(BASE_DIR, "tfidf.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

meta = {
    "metrics": metrics,
    "stats": {
        "total":       total_raw,
        "spam":        spam_raw,
        "ham":         ham_raw,
        "after_clean": len(df),
        "train":       int(X_tr.shape[0]),
        "test":        int(X_te.shape[0]),
    },
    "frequent_words": list(frequent_words),
    "rare_words":     list(rare_words),
}
with open(os.path.join(BASE_DIR, "meta.json"), "w") as f:
    json.dump(meta, f)

print("Saved: model.pkl  tfidf.pkl  meta.json")
print("Done.")
