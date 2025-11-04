# encode_tld.py
import pandas as pd, joblib, numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Encode TLDs in train dataset and save the encoder
"""

train = pd.read_csv("data/processed/train.csv").fillna(0)
test = pd.read_csv("data/processed/test.csv").fillna(0)

le = LabelEncoder()
train["tld_enc"] = le.fit_transform(train["tld"].astype(str))

# Add unknown token for unseen test TLDs
if "__unknown__" not in le.classes_:
    le.classes_ = np.append(le.classes_, "__unknown__")

joblib.dump(le, "model/tld_encoder.pkl")
print("[+] Saved TLD encoder with", len(le.classes_), "classes.")
"""

# Encode TLDs in test dataset using the saved encoder and append to datasets
# Load datasets
train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

print("[*] Loading train/test datasets ...")
train = pd.read_csv(train_path).fillna(0)
test = pd.read_csv(test_path).fillna(0)

# Encode TLDs using LabelEncoder
print("[*] Encoding TLDs ...")
le = LabelEncoder()
train["tld_enc"] = le.fit_transform(train["tld"].astype(str))

# Add '__unknown__' token for unseen TLDs in test set
if "__unknown__" not in le.classes_:
    le.classes_ = np.append(le.classes_, "__unknown__")

# Apply the same encoder to test data
known_classes = set(le.classes_)
test["tld_enc"] = [
    le.transform([t if t in known_classes else "__unknown__"])[0]
    for t in test["tld"].astype(str)
]

# Save the encoder
os.makedirs("model", exist_ok=True)
joblib.dump(le, "model/tld_encoder.pkl")
print(f"[+] Saved TLD encoder with {len(le.classes_)} classes → model/tld_encoder.pkl")

# Save updated datasets
train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

print(f"[+] Updated datasets saved:")
print(f"    → {train_path}")
print(f"    → {test_path}")
print("[✅] Now both train/test include 'tld_enc' column.")
