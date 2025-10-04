
# Random Forest Classifier for Phishing URL Detection
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from vis_metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall

# ===============================
# Load Dataset
# ===============================
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Drop non-numeric cols (keep only features)
X_train = train.drop(columns=["url", "label", "tld"])
y_train = train["label"]

X_test = test.drop(columns=["url", "label", "tld"])
y_test = test["label"]

# ===============================
# Train Random Forest
# ===============================
print("[*] Training RandomForest...")

clf = RandomForestClassifier(
    n_estimators=300,  # number of trees
    max_depth=None,  # let trees grow fully
    min_samples_split=2,
    n_jobs=-1,  # use all CPU cores
    random_state=42,
    class_weight="balanced",  # handle imbalance just in case
)

clf.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # probability estimates for the positive class

os.makedirs("plots", exist_ok=True)

plot_confusion_matrix(y_test, y_pred, save_path="plots/confusion_matrix.png")
plot_roc_curve(
    y_test, y_prob, model_name="Random Forest", save_path="plots/roc_curve.png"
)
plot_precision_recall(
    y_test, y_prob, model_name="Random Forest", save_path="plots/pr_curve.png"
)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# ===============================
# Save Model
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/random_forest.pkl")
print("\n[+] Model saved to models/random_forest.pkl")
