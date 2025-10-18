import os
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
# df = pd.read_csv("models/xai_eval_final.csv")
df = pd.read_csv("models/xai_eval_final.csv", low_memory=False)

# Drop rows with missing labels or probabilities
df = df.dropna(subset=["label", "hybrid_prob"])

# Convert label column to numeric in case it’s stored as string
df["label"] = df["label"].astype(int)


# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# ------------------------------
# 2️⃣ Evaluate multiple thresholds
# ------------------------------
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
report_data = []

for t in thresholds:
    preds = (df["hybrid_prob"] > t).astype(int)
    report = classification_report(df["label"], preds, digits=4, output_dict=True)

    # Store key metrics
    report_data.append(
        {
            "Threshold": t,
            "Accuracy": report["accuracy"],
            "Precision_Benign": report["0"]["precision"],
            "Recall_Benign": report["0"]["recall"],
            "F1_Benign": report["0"]["f1-score"],
            "Precision_Phishing": report["1"]["precision"],
            "Recall_Phishing": report["1"]["recall"],
            "F1_Phishing": report["1"]["f1-score"],
            "Macro_F1": report["macro avg"]["f1-score"],
        }
    )

    print(f"\n=== Threshold: {t} ===")
    print(classification_report(df["label"], preds, digits=4))

# Save all reports to CSV
report_df = pd.DataFrame(report_data)
report_path = "plots/threshold_reports.csv"
report_df.to_csv(report_path, index=False)
print(f"\n✅ Saved threshold performance reports to: {report_path}")

# ------------------------------
# 3️⃣ Plot F1-score curve
# ------------------------------
prec, rec, thres = precision_recall_curve(df["label"], df["hybrid_prob"])
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

plt.figure(figsize=(8, 5))
plt.plot(thres, f1[:-1], color="blue", linewidth=2)
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.title("Threshold Tuning Curve")
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = "plots/threshold_tuning.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"✅ Saved F1-score plot to: {plot_path}")

# ------------------------------
# 4️⃣ Find the best threshold
# ------------------------------
best_t = thres[f1.argmax()]
best_f1 = f1.max()
print(f"\n🏆 Best threshold: {best_t:.4f} (F1-score: {best_f1:.4f})")

# Save best threshold to text file
with open("plots/best_threshold.txt", "w") as f:
    f.write(f"Best threshold: {best_t:.4f}\nF1-score: {best_f1:.4f}")
print("✅ Saved best threshold to: plots/best_threshold.txt")
