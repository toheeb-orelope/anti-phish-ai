# =====================================
# LightGBM + Calibration (auto tld_enc)
# =====================================
import os, joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
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

# Prepare features and labels
X_train = train.drop(columns=["url", "label", "tld"], errors="ignore")
X_test = test.drop(columns=["url", "label", "tld"], errors="ignore")
y_train = train["label"]
y_test = test["label"]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Train + Calibrate LightGBM
print("[*] Training LightGBM with isotonic calibration ...")
base_lgbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    random_state=42,
    class_weight="balanced",
)
calibrated_lgbm = CalibratedClassifierCV(base_lgbm, method="isotonic", cv=5)
calibrated_lgbm.fit(X_train, y_train)
print("[+] Base LightGBM trained and calibrated successfully.")

# --- Evaluate
y_pred = calibrated_lgbm.predict(X_test)
y_prob = calibrated_lgbm.predict_proba(X_test)[:, 1]

os.makedirs("plot", exist_ok=True)
plot_confusion_matrix(y_test, y_pred, "plot/lgbm_confusion_matrix.png")
plot_roc_curve(y_test, y_prob, "LightGBM (Calibrated)", "plot/lgbm_roc_curve.png")
plot_precision_recall(y_test, y_prob, "LightGBM (Calibrated)", "plot/lgbm_pr_curve.png")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(calibrated_lgbm, "model/lgbm_calibrated.pkl")
print("[+] Calibrated LGBM saved to model/lgbm_calibrated.pkl")
