# LightGBM Classifier for Phishing URL Detection
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from vis_metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

# ===============================
# Load Dataset
# ===============================
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Drop non-numeric columns
X_train = train.drop(columns=["url", "label", "tld"])
y_train = train["label"]
X_test = test.drop(columns=["url", "label", "tld"])
y_test = test["label"]

# ===============================
# Train LightGBM
# ===============================
print("[*] Training LightGBM...")

clf = lgb.LGBMClassifier(
    n_estimators=400,  # number of boosting rounds
    learning_rate=0.05,  # shrinkage rate
    max_depth=-1,  # unlimited depth by default
    num_leaves=64,  # number of leaves per tree
    subsample=0.8,  # sample a fraction of data
    colsample_bytree=0.8,  # sample a fraction of features
    reg_lambda=1.0,  # L2 regularization
    min_child_samples=20,  # minimum samples per leaf
    objective="binary",  # binary classification
    boosting_type="gbdt",  # gradient boosting decision tree
    n_jobs=-1,  # use all CPU cores
    random_state=42,
)

clf.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

os.makedirs("plots", exist_ok=True)

plot_confusion_matrix(y_test, y_pred, save_path="plots/lgbm_confusion_matrix.png")
plot_roc_curve(
    y_test, y_prob, model_name="LightGBM", save_path="plots/lgbm_roc_curve.png"
)
plot_precision_recall(
    y_test, y_prob, model_name="LightGBM", save_path="plots/lgbm_pr_curve.png"
)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# ===============================
# Save Model
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/lightgbm_model.pkl")
print("\n[+] Model saved to models/lightgbm_model.pkl")
