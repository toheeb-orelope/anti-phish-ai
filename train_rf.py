# =====================================
# Random Forest + Calibration (auto tld_enc)
# =====================================
import os, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from vis_metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

print("[*] Loading processed datasets ...")
train = pd.read_csv("data/processed/train.csv").fillna(0)
test = pd.read_csv("data/processed/test.csv").fillna(0)

# ✅ Auto-encode TLDs if tld_enc missing
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

# --- Train + calibrate RF
base_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
base_rf.fit(X_train, y_train)
print("[+] Base Random Forest model trained successfully.")
print("[*] Calibrating probabilities (isotonic)...")
calibrated_rf = CalibratedClassifierCV(base_rf, method="isotonic", cv=5)
calibrated_rf.fit(X_train, y_train)

# --- Evaluate
y_pred = calibrated_rf.predict(X_test)
y_prob = calibrated_rf.predict_proba(X_test)[:, 1]

os.makedirs("plot", exist_ok=True)
plot_confusion_matrix(y_test, y_pred, "plot/rf_confusion_matrix.png")
plot_roc_curve(y_test, y_prob, "Random Forest (Calibrated)", "plot/rf_roc_curve.png")
plot_precision_recall(
    y_test, y_prob, "Random Forest (Calibrated)", "plot/rf_pr_curve.png"
)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(calibrated_rf, "model/rf_calibrated.pkl")
print("[+] Calibrated RF saved to model/rf_calibrated.pkl")
