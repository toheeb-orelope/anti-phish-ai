# evaluate_threshold.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# --------------------------------------------------
# STEP 1: Load your test dataset and predictions
# --------------------------------------------------
# Assumes you have:
#  - data/processed/test.csv  → contains 'label' column (0 = benign, 1 = phishing)
#  - models/ensemble_probs.npy → contains predicted probabilities from your ensemble

test = pd.read_csv("data/processed/test.csv")
y_true = test["label"].values
# y_pred = np.load("models/ensemble_probs.npy")  # predicted probabilities (0–1)
probs = np.load("models/ensemble_probs.npy", allow_pickle=True)

# Average across models (axis=1) to get one ensemble score per sample
if probs.ndim > 1:
    y_pred = probs.mean(axis=1)
else:
    y_pred = probs


print("🔍 Checking predictions file...")
print("Shape:", y_pred.shape)
print("Min:", y_pred.min(), "Max:", y_pred.max())
print("Unique values:", np.unique(y_pred)[:10])


# --------------------------------------------------
# STEP 2: Compute ROC curve and AUC
# --------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# --------------------------------------------------
# STEP 3: Find the optimal threshold using Youden’s J statistic
# --------------------------------------------------
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# --------------------------------------------------
# STEP 4: Print results
# --------------------------------------------------
print("✅ Optimal threshold:", round(optimal_threshold, 4))
print("✅ ROC AUC:", round(roc_auc, 4))

# --------------------------------------------------
# STEP 5: Optional – Save threshold for use in xai_explain.py
# --------------------------------------------------
with open("models/optimal_threshold.txt", "w") as f:
    f.write(str(optimal_threshold))

print("✅ Threshold saved to models/optimal_threshold.txt")

# --------------------------------------------------
# Model references (not used directly here)
# --------------------------------------------------
rf_model = "models/random_forest.pkl"
xgb_model = "models/xgboost_model.pkl"
lgbm_model = "models/lightgbm_model.pkl"
cnn_model = "models/cnn_best.ckpt"
lstm_model = "models/lstm_best.ckpt"
ffnn_model = "models/ffnn_best.ckpt"
