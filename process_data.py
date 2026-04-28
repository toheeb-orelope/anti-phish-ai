import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
from extract_features import extract_features  # your existing feature builder

# ========== Load datasets ==========
phiusiil = pd.read_csv("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
phishtank = pd.read_csv("data/raw/verified_online.csv")
malicious = pd.read_csv("data/raw/malicious_phish.csv")
alexa = pd.read_csv("data/raw/top-1m.csv", header=None, names=["rank", "domain"])

# ========== Standardize format ==========
phiusiil_clean = phiusiil[["URL", "label"]].rename(columns={"URL": "url"})

phishtank_clean = phishtank[["url"]].copy()
phishtank_clean["label"] = 1

malicious_clean = malicious.copy()
malicious_clean["label"] = malicious_clean["type"].apply(
    lambda x: 0 if x == "benign" else 1
)
malicious_clean = malicious_clean[["url", "label"]]

alexa_clean = alexa.copy()
alexa_clean["url"] = "http://" + alexa_clean["domain"]
alexa_clean["label"] = 0
alexa_clean = alexa_clean[["url", "label"]]

# ========== Merge all ==========
all_data = pd.concat(
    [phiusiil_clean, phishtank_clean, malicious_clean, alexa_clean], ignore_index=True
)

# Normalize
all_data["url"] = all_data["url"].astype(str).str.lower().str.strip()
all_data.drop_duplicates(subset=["url"], inplace=True)


print("[*] Extracting features (Lexical + SSL + DNS + WHOIS)... this may take a while.")
records = [extract_features(u) for u in all_data["url"]]
features_df = pd.DataFrame(records)
processed = pd.concat(
    [all_data.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
)

# features_df = all_data["url"].apply(lambda x: pd.Series(extract_features(x)))
processed = pd.concat([all_data, features_df], axis=1)

# ========== Balance the dataset ==========
phish_data = processed[processed["label"] == 1]
benign_data = processed[processed["label"] == 0]

benign_downsampled = resample(
    benign_data, replace=False, n_samples=phish_data.shape[0], random_state=42
)

balanced = pd.concat([phish_data, benign_downsampled], ignore_index=True)
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(
    f"Balanced dataset → Phish: {balanced[balanced['label']==1].shape[0]}, "
    f"Benign: {balanced[balanced['label']==0].shape[0]}"
)

# ========== Train/Test Split ==========
train, test = train_test_split(
    balanced, test_size=0.2, stratify=balanced["label"], random_state=42
)

# Create processed directory
os.makedirs("data/processed", exist_ok=True)

# Save
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("✅ Saved:")
print("   data/processed/train.csv")
print("   data/processed/test.csv")
print(f"   Train size: {train.shape}, Test size: {test.shape}")
