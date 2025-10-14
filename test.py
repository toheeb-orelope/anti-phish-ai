import numpy as np

# Load the file
# data = np.load("models/ensemble_probs.npy", allow_pickle=True)

# print(type(data))
# print(data)
from phishin_train_cnn import LightningCNN
import torch

model = LightningCNN.load_from_checkpoint(
    "models/cnn_lightning.ckpt", map_location="cpu"
)
model.eval()
print("✅ CNN loaded successfully")


# Retrain LightGBM with numeric 'tld' column
"""
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# -------------------------------
# Step 1: Load train and test datasets
# -------------------------------
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_question",
    "num_equals",
    "num_digits",
    "num_subdirs",
    "has_https",
    "tld",
]

X_train = train[FEATURE_COLUMNS].copy()
y_train = train["label"]
X_test = test[FEATURE_COLUMNS].copy()
y_test = test["label"]

# -------------------------------
# Step 2: Encode TLD numerically (same as before)
# -------------------------------
from sklearn.preprocessing import LabelEncoder

print("🔹 Encoding 'tld' column as numeric...")
label_encoder = LabelEncoder()
all_tlds = pd.concat([X_train["tld"], X_test["tld"]]).astype(str)
label_encoder.fit(all_tlds)

X_train["tld"] = label_encoder.transform(X_train["tld"].astype(str))
X_test["tld"] = label_encoder.transform(X_test["tld"].astype(str))

# Save encoder for consistency
joblib.dump(label_encoder, "models/tld_encoder.pkl")

# -------------------------------
# Step 3: Train LightGBM (pure numeric features)
# -------------------------------
print("🔹 Training LightGBM with numeric TLD...")
lgbm = LGBMClassifier(random_state=42, verbose=-1)

# Ensure pure numeric data
X_train = X_train.astype(float)
X_test = X_test.astype(float)

lgbm.fit(X_train, y_train)

# Save the retrained model
joblib.dump(lgbm, "models/lightgbm_model1.pkl")
print("✅ Saved models/lightgbm_model.pkl")

# -------------------------------
# Step 4: Evaluate
# -------------------------------
y_pred = lgbm.predict(X_test)
y_prob = lgbm.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n📊 Evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ LightGBM retrained successfully with numeric 'tld' column.")
"""


import pandas as pd
from run_xai import run_example

test = pd.read_csv("data/processed/test.csv")

for url in test["url"].head(200):  # or full dataset
    try:
        run_example(url)
    except Exception as e:
        print("Error:", e)
