from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib, os
from vis_metrics import plot_confusion_matrix

# Load dataset
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train = train.drop(columns=["url", "label", "tld"])
y_train = train["label"]
X_test = test.drop(columns=["url", "label", "tld"])
y_test = test["label"]

print("[*] Training LinearSVC...")

clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Plot confusion matrix
os.makedirs("plots", exist_ok=True)
plot_confusion_matrix(y_test, y_pred, save_path="plots/svm_linear_confusion_matrix.png")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/svm_linear.pkl")
print("\n[+] Model saved to models/svm_linear.pkl")
