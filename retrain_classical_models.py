import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
import os

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
# Step 2: Encode the TLD column (string → number)
# -------------------------------
print("🔹 Encoding 'tld' feature...")

label_encoder = LabelEncoder()
all_tlds = pd.concat([X_train["tld"], X_test["tld"]]).astype(str)
label_encoder.fit(all_tlds)

X_train["tld"] = label_encoder.transform(X_train["tld"].astype(str))
X_test["tld"] = label_encoder.transform(X_test["tld"].astype(str))

# Optional: Save encoder for later use (so new URLs can be encoded consistently)
joblib.dump(label_encoder, "models/tld_encoder.pkl")

# -------------------------------
# Step 3: Train models
# -------------------------------
print("🔹 Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")
print("✅ Saved models/random_forest.pkl")

print("🔹 Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "models/xgboost_model.pkl")
print("✅ Saved models/xgboost_model.pkl")

print("🔹 Training LightGBM...")
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
joblib.dump(lgbm, "models/lightgbm_model.pkl")
print("✅ Saved models/lightgbm_model.pkl")

# -------------------------------
# Step 4: Evaluate models on test set
# -------------------------------
results = []

for name, model in [("RandomForest", rf), ("XGBoost", xgb), ("LightGBM", lgbm)]:
    print(f"\n📊 Evaluating {name}...")

    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else np.zeros_like(y_pred)
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append(
        {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc, 4),
            "True_Negative": tn,
            "False_Positive": fp,
            "False_Negative": fn,
            "True_Positive": tp,
        }
    )

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} ROC AUC: {roc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save ROC curve points
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    np.save(f"models/{name.lower()}_roc.npy", {"fpr": fpr, "tpr": tpr})

# -------------------------------
# Step 5: Save metrics to CSV
# -------------------------------
os.makedirs("models", exist_ok=True)
df_results = pd.DataFrame(results)
df_results.to_csv("models/model_performance.csv", index=False)
print("\n✅ Saved metrics to models/model_performance.csv")

# -------------------------------
# Step 6: Summary
# -------------------------------
print("\nSummary of Model Performance:")
print(df_results)

print("\n✅ Feature alignment check:")
print("RandomForest features:", rf.feature_names_in_)
print("XGBoost features:", xgb.feature_names_in_)
print("LightGBM features:", lgbm.feature_name_)
