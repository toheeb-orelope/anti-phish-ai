# =====================================
# XGBoost + Logistic Calibration (auto tld_enc)
# =====================================
import os, joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from vis_metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

print("[*] Loading processed datasets ...")
train = pd.read_csv("data/processed/train.csv").fillna(0)
test = pd.read_csv("data/processed/test.csv").fillna(0)

# ✅ Auto-encode TLDs if missing
if "tld_enc" not in train.columns:
    print("[INFO] 'tld_enc' not found — encoding using model/tld_encoder.pkl")
    le_path = "model/tld_encoder.pkl"
    if not os.path.exists(le_path):
        raise FileNotFoundError(
            "Please run encode_tld.py first to generate tld_encoder.pkl"
        )

    le = joblib.load(le_path)
    known_classes = set(le.classes_)

    train["tld_enc"] = [
        le.transform([t if t in known_classes else "__unknown__"])[0]
        for t in train["tld"].astype(str)
    ]
    test["tld_enc"] = [
        le.transform([t if t in known_classes else "__unknown__"])[0]
        for t in test["tld"].astype(str)
    ]

# Prepare features / labels
X_train = train.drop(columns=["url", "label", "tld"], errors="ignore")
X_test = test.drop(columns=["url", "label", "tld"], errors="ignore")
y_train = train["label"]
y_test = test["label"]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Train base XGBoost
print("[*] Training XGBoost base model ...")
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
print("[+] Base XGBoost model trained successfully.")

# --- Calibrate using Logistic Regression
print("[*] Calibrating probabilities (Logistic)...")
train_probs = xgb_model.predict_proba(X_train)[:, 1]
test_probs = xgb_model.predict_proba(X_test)[:, 1]

calibrator = LogisticRegression(max_iter=1000)
calibrator.fit(train_probs.reshape(-1, 1), y_train)

calibrated_probs = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
y_pred = (calibrated_probs > 0.5).astype(int)

# --- Evaluate
os.makedirs("plot", exist_ok=True)
plot_confusion_matrix(y_test, y_pred, "plot/xgb_confusion_matrix.png")
plot_roc_curve(
    y_test, calibrated_probs, "XGBoost (Calibrated)", "plot/xgb_roc_curve.png"
)
plot_precision_recall(
    y_test, calibrated_probs, "XGBoost (Calibrated)", "plot/xgb_pr_curve.png"
)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# --- Save model and calibrator
os.makedirs("model", exist_ok=True)
joblib.dump({"xgb": xgb_model, "calibrator": calibrator}, "model/xgb_calibrated.pkl")
print("[+] Calibrated XGBoost saved to model/xgb_calibrated.pkl")
